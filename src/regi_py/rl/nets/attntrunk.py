"""AttnTrunkNet -- BasicNet with global card-to-card self-attention in the trunk.

The smallest architectural delta over ``BasicNet`` (roadmap Option A): identical
inputs (``location`` / ``used_pile`` / ``capability`` frames-as-channels), identical
value / keepyness / action heads, and the identical ``CombineNet`` conv trunk on the
56x22 grid. The ONE change is a residual ``CardSelfAttention`` block inserted after
the trunk, giving the net the global, content-based card-to-card interaction the
pure-conv ``BasicNet`` lacks (its convs mix cards only locally along the bogus card
axis, and ``WidthCrossAttention`` mixes only *within* a card).

Because inputs, heads, targets and training layout are unchanged, this subclasses
``BasicNet`` directly and inherits ``_assemble`` / ``TRAIN_FIELDS`` / ``max_history``
/ ``tensorify_*`` / ``predict`` / ``calculate_loss`` -- only ``__init__`` (add the
attention block) and ``forward`` (apply it) differ.
"""
from regi_py.rl.nets.basicnet import BasicNet
from regi_py.rl.subnets import CardSelfAttention, _norm_groups

import torch.nn as nn

# the CombineNet trunk emits a 64-channel map on the 56x22 grid
_TRUNK_CHANNELS = 64


class AttnTrunkNet(BasicNet):
    __mname__ = "attntrunk"

    def __init__(self):
        super().__init__()
        # global card-to-card mixing on the 64-ch trunk; heads=8 divides 64.
        # GroupNorm keeps single-sample predict() consistent with training, like
        # the conv blocks (no BatchNorm running-stat train/infer mismatch).
        self.card_attn = CardSelfAttention(channels=_TRUNK_CHANNELS, heads=8)
        self.attn_norm = nn.GroupNorm(
            num_groups=_norm_groups(_TRUNK_CHANNELS),
            num_channels=_TRUNK_CHANNELS,
        )

    def forward(self, data):
        x1 = self.usp_net(data["used_pile"])
        x2 = self.loc_net(data["location"])
        x3 = self.cap_net(data["capability"])
        x = self.combiner(x1, x2, x3)              # (N, 64, 56, 22)
        # residual global card mixing, leaving the trunk shape unchanged so the
        # existing value / keepy / action heads consume it exactly as in BasicNet
        x = x + self.attn_norm(self.card_attn(x))
        v = self.v_net(x)
        k = self.k_net(x)
        a = self.a_net(x, k)
        return v, k, a
