"""PerCardMLPNet -- per-card MLP with NO cross-card mixing (the control net).

Roadmap baseline. Uses the same card-token input as ``CardTransformerNet``
(``features.fuse_card_tokens``) and the same shared per-card heads, but the trunk
is a plain MLP applied independently to each card token -- there is no attention,
no mixing, nothing that lets one card's features influence another's. It exists to
answer "is the cross-card interaction (attention / mixing) actually buying anything
over a net that sees each card in isolation?" -- a fair A/B control for
``CardTransformerNet`` / ``MixerNet`` since input, heads and capacity are matched.
"""
from regi_py.rl import features
from regi_py.rl.az.nets.base import BaseNet
from regi_py.rl.subnets import PerCardHeads

import torch
import torch.nn as nn

DIM = 64


class PerCardMLPNet(BaseNet):
    __mname__ = "percardmlp"
    max_history = 8
    TRAIN_FIELDS = ("tokens", "value", "keepyness", "atk_probs", "attacking")

    def __init__(self):
        super().__init__()
        in_dim = self.max_history * features.FEATURE_WIDTH
        # nn.Linear runs over the last (feature) axis only, so this whole trunk is
        # applied per card token independently -- no information crosses cards
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
        )
        self.heads = PerCardHeads(DIM)

    @classmethod
    def _assemble(cls, loc, usp, cap):
        tok = features.fuse_card_tokens(loc, usp, cap)  # (56, window*FEATURE_WIDTH)
        return {"tokens": torch.from_numpy(tok).unsqueeze(0)}

    def forward(self, data):
        x = self.mlp(data["tokens"])   # (N, 56, DIM), per-card
        return self.heads(x)
