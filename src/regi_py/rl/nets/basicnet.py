"""BasicNet -- the original AlphaZero architecture, now a ``BaseNet`` subclass.

Behavior is identical to the pre-refactor ``rl/basicnet.py``: three per-card input
streams (location / used_pile / capability) with the ``max_history`` window stacked
as the conv channel axis, a cross-attention ``CombineNet`` trunk on the 56x22 grid,
and value / keepyness / action heads. The shared featurization, tensorify, predict
and loss now live in ``features`` / ``nets.base``; this file keeps only the layout
(``_assemble``), the trunk/heads, and ``forward``.
"""
from regi_py.core import ComboTable
from regi_py.core import MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS
from regi_py.rl.nets.base import BaseNet
from regi_py.rl.subnets import (
    Conv1dBlock,
    Conv2dBlock,
    WidthCrossAttention,
)

import numpy as np
import torch
import torch.nn as nn

# flattened trunk width for the value/keepy heads: the trunk lives on the
# used-pile grid (card axis x played-status axis), so 56 x 22
_TRUNK_FLAT = MAX_CARDS_IN_GAME * int(MAX_PLAYED_STATUS)


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = Conv2dBlock(channels=(64, 8, 1), shapes=(3, 1), paddings=(1, 0))
        self.net2 = nn.Linear(in_features=_TRUNK_FLAT, out_features=1)
        self.ac = nn.Tanh()

    def forward(self, x):
        x = self.net1(x).reshape(x.shape[0], -1)
        x = self.net2(x)
        x = self.ac(x)
        return x


class KeepyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = Conv2dBlock(channels=(64, 8, 1), shapes=(3, 1), paddings=(1, 0))
        self.net2 = nn.Linear(in_features=_TRUNK_FLAT, out_features=MAX_CARDS_IN_GAME)
        self.ac = nn.Sigmoid()

    def forward(self, x):
        x = self.net1(x).reshape(x.shape[0], -1)
        x = self.net2(x)
        x = self.ac(x)
        return x


class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        n = MAX_CARDS_IN_GAME
        # trunk (N, 64, 56, 22) -> one logit per (card, played_status) cell, with
        # the 56 x 22 grid preserved throughout
        self.net1 = Conv2dBlock(channels=(64, 16, 1), shapes=(3, 1), paddings=(1, 0))
        # mix along the played-status axis (56 cards as channels), width preserved
        self.net2 = Conv1dBlock(channels=(n, n, n), shapes=(3, 3), paddings=(1, 1))
        # fuse per-card keepyness into the per-card action logits; 56 % 8 == 0
        self.wca = WidthCrossAttention(channels=MAX_CARDS_IN_GAME, heads=8)
        # additive softmax mask over the flattened (56 x 22) action grid: 0 on
        # structurally-valid (location, played_status) cells, -inf on impossible
        # ones. Built once here so the softmax just adds it (no per-forward fill).
        valid = np.array(ComboTable.all_entries(), dtype=np.float32)  # (56, 22)
        add_mask = np.where(valid == 0, -np.inf, 0.0).astype(np.float32).reshape(1, -1)
        self.register_buffer("invalid_mask", torch.from_numpy(add_mask))

    def forward(self, x0, k):
        x = self.net1(x0)  # (N, 1, 56, 22)
        x = x.reshape(x0.shape[0], MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)  # (N, 56, 22)
        x = self.net2(x)  # (N, 56, 22)
        x = x.reshape(x.shape[0], MAX_CARDS_IN_GAME, 1, MAX_PLAYED_STATUS)
        k2 = k.reshape(x.shape[0], MAX_CARDS_IN_GAME, 1, 1)
        logits = self.wca(x, k2).reshape(-1, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)
        # masked softmax over the whole (56 x 22) action grid
        n = logits.shape[0]
        flat = torch.softmax(logits.reshape(n, -1) + self.invalid_mask, dim=-1)
        return flat.reshape(-1, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)


class CombineNet(nn.Module):
    def __init__(self, channels=32, reduction=4):
        super().__init__()
        self.wca1 = WidthCrossAttention(channels=channels, heads=4)
        self.wca2 = WidthCrossAttention(channels=channels, heads=4)
        self.net = Conv2dBlock(
            channels=(channels, 64, 64, 64, 64, 64, 64, 64, 64, 64),
            shapes=(1, 3, 3, 1, 3, 3, 1, 3, 3),
            paddings=(0, 1, 1, 0, 1, 1, 0, 1, 1),
        )

    def forward(self, usp, loc, cap):
        y1 = self.wca1(usp, loc)
        y2 = self.wca2(y1, cap)
        y3 = self.net(y2)
        return y3


class BasicNet(BaseNet):
    __mname__ = "basic"
    # history window length; a class attribute so callers (e.g. the brute
    # recorder in rl.training) can read it without instantiating the net
    max_history = 8

    # single source of truth for the training tensor field order, shared by
    # tensorify_training / ShardBuffer / run_epoch
    TRAIN_FIELDS = (
        "location",
        "used_pile",
        "capability",
        "value",
        "keepyness",
        "atk_probs",
        "attacking",
    )

    def __init__(self):
        super().__init__()
        #
        self.loc_net = Conv2dBlock(
            channels=(self.max_history, 32),
            shapes=(1,),
            paddings=(0,),
        )
        self.usp_net = Conv2dBlock(
            channels=(self.max_history, 32),
            shapes=(1,),
            paddings=(0,),
        )
        self.cap_net = Conv2dBlock(
            channels=(self.max_history, 32),
            shapes=(1,),
            paddings=(0,),
        )

        self.combiner = CombineNet(channels=32)
        self.v_net = ValueNet()
        self.k_net = KeepyNet()
        self.a_net = ActionNet()

    @classmethod
    def _assemble(cls, loc, usp, cap):
        # frames-as-channels: each raw (window, 56, W) array becomes the model
        # input (1, window, 56, W); the first conv's in-channels == max_history
        return {
            "location": torch.from_numpy(loc).unsqueeze(0),
            "used_pile": torch.from_numpy(usp).unsqueeze(0),
            "capability": torch.from_numpy(cap).unsqueeze(0),
        }

    def forward(self, data):
        x1 = self.usp_net(data["used_pile"])
        x2 = self.loc_net(data["location"])
        x3 = self.cap_net(data["capability"])
        x = self.combiner(x1, x2, x3)
        v = self.v_net(x)
        k = self.k_net(x)
        a = self.a_net(x, k)
        return v, k, a
