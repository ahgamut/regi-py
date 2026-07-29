"""MixerNet -- MLP-Mixer over card tokens (static global card-to-card mixing).

Roadmap architecture. Same card-token input as ``CardTransformerNet`` and the same
shared per-card heads, but the trunk is a stack of MLP-Mixer blocks. Each block's
token-mixing MLP mixes information across the 56 cards through a *static, learned*
weight matrix (every position pair has a fixed learned interaction), in contrast to
the *content-dependent* card-to-card interaction a Transformer computes from the
tokens themselves. It is the A/B partner for ``CardTransformerNet``: does the mixing
need to be attention (data-dependent), or does a fixed learned card-mixing matrix
suffice on this fixed 56-card set?

The window frames are flattened into each card token's feature vector (the shared
``features.fuse_card_tokens`` layout) rather than kept as a separate mixed axis --
the essential inductive bias here is the static global card mixing.
"""
from regi_py.core import MAX_CARDS_IN_GAME
from regi_py.rl import features
from regi_py.rl.nets.base import BaseNet
from regi_py.rl.subnets import MixerBlock, PerCardHeads

import torch
import torch.nn as nn

DIM = 64
N_BLOCKS = 4
TOKEN_HIDDEN = 128
CHANNEL_HIDDEN = 128


class MixerNet(BaseNet):
    __mname__ = "mixer"
    max_history = 8
    TRAIN_FIELDS = ("tokens", "value", "keepyness", "atk_probs", "attacking")

    def __init__(self):
        super().__init__()
        in_dim = self.max_history * features.FEATURE_WIDTH
        self.proj = nn.Linear(in_dim, DIM)
        self.blocks = nn.ModuleList(
            MixerBlock(MAX_CARDS_IN_GAME, DIM, TOKEN_HIDDEN, CHANNEL_HIDDEN)
            for _ in range(N_BLOCKS)
        )
        self.heads = PerCardHeads(DIM)

    @classmethod
    def _assemble(cls, loc, usp, cap):
        tok = features.fuse_card_tokens(loc, usp, cap)  # (56, window*FEATURE_WIDTH)
        return {"tokens": torch.from_numpy(tok).unsqueeze(0)}

    def forward(self, data):
        x = self.proj(data["tokens"])   # (N, 56, DIM)
        for block in self.blocks:
            x = block(x)
        return self.heads(x)
