"""MultiHotActionNet (``"adzmulti"``) -- the first ADZ candidate-scoring net.

DouZero-literal action encoding: every offered subset is a multi-hot vector over
the 56 card locations (its member cards) concatenated with its semantic feature
row, encoded to a key; the state is one contextual token per card (a Transformer
over the card tokens, same building blocks as ``cardtx``) pooled to a query. The
score of a candidate is the scaled dot product of the state query with its key,
masked-softmaxed over the phase's real offered subsets. Value is a card-space head
on the pooled state; keepyness is a per-card sigmoid head on the contextual card
embeddings (the derived-CFR aux, regularizing the trunk).

Uses ``dropout=0.0`` so single-sample self-play ``predict`` behaves identically to
training (LayerNorm is already train/eval invariant).
"""
import math

from regi_py.core import MAX_CARDS_IN_GAME
from regi_py.rl import features
from regi_py.rl.adz_nets import register_adz
from regi_py.rl.adz_nets.base import CandidateBaseNet
from regi_py.rl.subnets import MultiHotActionEncoder

import numpy as np
import torch
import torch.nn as nn

D_MODEL = 64
N_HEADS = 8
N_LAYERS = 3
FF_DIM = 128


@register_adz
class MultiHotActionNet(CandidateBaseNet):
    __mname__ = "adzmulti"
    max_history = 8
    TRAIN_FIELDS = (
        "tokens",
        "cand_members",
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
        # ---- state encoder: one contextual token per card (like cardtx) ----
        in_dim = self.max_history * features.FEATURE_WIDTH
        self.embed = nn.Linear(in_dim, D_MODEL)
        # per-card identity positional embedding (which card each token is)
        self.card_emb = nn.Parameter(torch.randn(1, MAX_CARDS_IN_GAME, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        # ---- action encoder + heads ----
        self.action_enc = MultiHotActionEncoder(self.CAND_FEATURE_DIM, D_MODEL)
        self.value_head = nn.Linear(D_MODEL, 1)
        self.keepy_head = nn.Linear(D_MODEL, 1)

    @classmethod
    def _assemble_membership(cls, padded_bitwises):
        # multi-hot card membership from each candidate's bitwise set bits; a padded
        # (bitwise == 0) row is all-zero and is masked out by cand_mask anyway
        MC = cls.MAX_CANDIDATES
        members = np.zeros((MC, MAX_CARDS_IN_GAME), dtype=np.float32)
        for i, bw in enumerate(padded_bitwises):
            b = int(bw)
            while b:
                loc = (b & -b).bit_length() - 1  # index of the lowest set bit
                members[i, loc] = 1.0
                b &= b - 1
        return {"cand_members": torch.from_numpy(members).unsqueeze(0)}

    def forward(self, data):
        # state: contextual card embeddings + pooled query
        x = self.embed(data["tokens"]) + self.card_emb   # (N, 56, D)
        x = self.encoder(x)                              # (N, 56, D)
        q = x.mean(dim=1)                                # (N, D) pooled state query
        # action: one key per candidate, scored against the state query
        k = self.action_enc(data["cand_members"], data["cand_feats"])  # (N, K, D)
        logits = torch.einsum("nd,nkd->nk", q, k) / math.sqrt(self.d_model)  # (N, K)
        mask = data["cand_mask"]                         # (N, K); 1 real, 0 pad
        cand_logits = logits.masked_fill(mask == 0, float("-inf"))
        # heads
        value = torch.tanh(self.value_head(q))                 # (N, 1)
        keepy = torch.sigmoid(self.keepy_head(x).squeeze(-1))  # (N, 56)
        return value, cand_logits, keepy
