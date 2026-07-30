"""PooledActionNet (``"adzpool"``) -- the second ADZ candidate encoding.

The invariant check for the node/base abstraction: a DIFFERENT action encoding
that plugs in with ONLY this new file + a registry line -- ``ADZNode``,
``ADZNodeInfo``, ``adz.explorer``, ``CandidateBaseNet`` and ``subnets`` are all
untouched. Where ``adzmulti`` encodes a subset as a multi-hot(56) card vector fed
through an MLP (the cards' identities, not their state), ``adzpool`` encodes a
subset as the index+mask of its member card LOCATIONS and pools the STATE encoder's
own contextual card embeddings for those members (DouZero's index-style action
encoding) -- so the action key is grounded in the same per-card representation the
value/keepy heads read, not a fresh embedding table.

State encoder + heads are identical to ``adzmulti`` (a Transformer over the card
tokens, dropout 0 so predict == training); only ``_assemble_membership`` (index+mask
instead of multi-hot) and the action encoder (pool-and-project instead of
multi-hot-MLP) differ, plus the swapped membership columns in ``TRAIN_FIELDS``.
"""
import math

from regi_py.core import MAX_CARDS_IN_GAME
from regi_py.rl import features
from regi_py.rl.adz.nets import register_adz
from regi_py.rl.adz.nets.base import CandidateBaseNet

import numpy as np
import torch
import torch.nn as nn

D_MODEL = 64
N_HEADS = 8
N_LAYERS = 3
FF_DIM = 128
# max member cards in an offered subset = max hand size (a 7-card defense subset);
# verified against the built engine as the largest set-bit count over real offers
MAX_PARTS = 7


@register_adz
class PooledActionNet(CandidateBaseNet):
    __mname__ = "adzpool"
    max_history = 8
    TRAIN_FIELDS = (
        "tokens",
        "cand_idx",
        "cand_partmask",
        "cand_feats",
        "cand_mask",
        "policy",
        "value",
        "keepyness",
        "attacking",
    )

    def __init__(self):
        super().__init__()
        self.d_model = D_MODEL
        # ---- state encoder: one contextual token per card (same as adzmulti) ----
        in_dim = self.max_history * features.FEATURE_WIDTH
        self.embed = nn.Linear(in_dim, D_MODEL)
        self.card_emb = nn.Parameter(torch.randn(1, MAX_CARDS_IN_GAME, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        # ---- action encoder: pooled member embeddings + semantics -> key ----
        self.action_mlp = nn.Sequential(
            nn.Linear(D_MODEL + self.CAND_FEATURE_DIM, FF_DIM),
            nn.ReLU(),
            nn.Linear(FF_DIM, D_MODEL),
        )
        # ---- heads ----
        self.value_head = nn.Linear(D_MODEL, 1)
        self.keepy_head = nn.Linear(D_MODEL, 1)

    @classmethod
    def _assemble_membership(cls, padded_bitwises):
        # index+mask: each candidate's member card LOCATIONS (set bits of its
        # bitwise), zero-padded to MAX_PARTS, with a 1/0 mask. A padded candidate
        # or the yield combo (bitwise 0) has an all-zero mask -> pooled to 0.
        MC = cls.MAX_CANDIDATES
        idx = np.zeros((MC, MAX_PARTS), dtype=np.int64)
        partmask = np.zeros((MC, MAX_PARTS), dtype=np.float32)
        for i, bw in enumerate(padded_bitwises):
            b = int(bw)
            j = 0
            while b and j < MAX_PARTS:
                loc = (b & -b).bit_length() - 1  # index of the lowest set bit
                idx[i, j] = loc
                partmask[i, j] = 1.0
                b &= b - 1
                j += 1
        return {
            "cand_idx": torch.from_numpy(idx).unsqueeze(0),
            "cand_partmask": torch.from_numpy(partmask).unsqueeze(0),
        }

    def forward(self, data):
        # state: contextual card embeddings + pooled query (same as adzmulti)
        x = self.embed(data["tokens"]) + self.card_emb   # (N, 56, D)
        x = self.encoder(x)                              # (N, 56, D)
        q = x.mean(dim=1)                                # (N, D) pooled state query
        # action: pool each candidate's member cards' contextual embeddings
        cand_idx = data["cand_idx"]                      # (N, K, P) long
        partmask = data["cand_partmask"]                 # (N, K, P)
        N, S, D = x.shape
        K, P = cand_idx.shape[1], cand_idx.shape[2]
        flat = cand_idx.reshape(N, K * P)                # (N, K*P)
        gathered = torch.gather(x, 1, flat.unsqueeze(-1).expand(-1, -1, D))
        gathered = gathered.reshape(N, K, P, D)          # (N, K, P, D)
        m = partmask.unsqueeze(-1)                       # (N, K, P, 1)
        summed = (gathered * m).sum(dim=2)               # (N, K, D)
        denom = m.sum(dim=2).clamp_min(1.0)              # (N, K, 1); yield has 0 parts
        pooled = summed / denom                          # (N, K, D) masked mean
        feats = data["cand_feats"]                       # (N, K, F)
        k = self.action_mlp(torch.cat([pooled, feats], dim=-1))  # (N, K, D)
        logits = torch.einsum("nd,nkd->nk", q, k) / math.sqrt(self.d_model)
        cand_logits = logits.masked_fill(data["cand_mask"] == 0, float("-inf"))
        # heads
        value = torch.tanh(self.value_head(q))                 # (N, 1)
        keepy = torch.sigmoid(self.keepy_head(x).squeeze(-1))  # (N, 56)
        return value, cand_logits, keepy
