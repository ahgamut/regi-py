import torch
import torch.nn as nn


def _norm_groups(channels, target=8):
    """Largest group count <= target that divides ``channels`` (so GroupNorm is
    valid for every channel width used here: 1, 8, 16, 32, 56, 64)."""
    g = min(target, channels)
    while channels % g != 0:
        g -= 1
    return g


class LinearBlock(nn.Module):
    def __init__(self, shapes):
        super(LinearBlock, self).__init__()
        self.ac = nn.ReLU()
        # nn.ModuleList so the layers are registered params (a plain list is not)
        self.nets = nn.ModuleList(
            nn.Linear(in_features=shapes[i], out_features=shapes[i + 1])
            for i in range(len(shapes) - 1)
        )

    def forward(self, x0):
        x = x0
        for net in self.nets:
            y = self.ac(net(x))
            if x.shape == y.shape:
                x = x + y
            else:
                x = y
        return x


class Conv1dBlock(nn.Module):
    def __init__(self, shapes, channels, paddings):
        super(Conv1dBlock, self).__init__()
        self.ac = nn.ReLU()
        # nn.ModuleList so the conv/norm layers are registered params. GroupNorm
        # (not BatchNorm) so the single-sample predict() the CPU explorers/eval run
        # normalizes identically to training -- no running-stat train/infer mismatch.
        self.nets = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    kernel_size=shapes[i],
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    padding=paddings[i],
                    bias=False,
                ),
                nn.GroupNorm(
                    num_groups=_norm_groups(channels[i + 1]),
                    num_channels=channels[i + 1],
                ),
            )
            for i in range(len(channels) - 1)
        )

    def forward(self, x0):
        x = x0
        for net in self.nets:
            y = self.ac(net(x))
            if y.shape == x.shape:
                x = x + y
            else:
                x = y
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, shapes, channels, paddings):
        super(Conv2dBlock, self).__init__()
        self.ac = nn.ReLU()
        # nn.ModuleList so the conv/norm layers are registered params. GroupNorm
        # (not BatchNorm) so the single-sample predict() the CPU explorers/eval run
        # normalizes identically to training -- no running-stat train/infer mismatch.
        self.nets = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    kernel_size=shapes[i],
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    padding=paddings[i],
                    bias=False,
                ),
                nn.GroupNorm(
                    num_groups=_norm_groups(channels[i + 1]),
                    num_channels=channels[i + 1],
                ),
            )
            for i in range(len(channels) - 1)
        )

    def forward(self, x0):
        x = x0
        for net in self.nets:
            y = self.ac(net(x))
            if y.shape == x.shape:
                x = x + y
            else:
                x = y
        return x


class WidthCrossAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        assert channels % heads == 0
        self.h = heads
        self.dk = channels // heads
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, a, b):
        # a: (N, C, H, y1)  b: (N, C, H, y2)  -> (N, C, H, y1)
        N, C, H, y1 = a.shape
        y2 = b.shape[-1]

        a = a.permute(0, 2, 3, 1).reshape(N * H, y1, C)  # queries
        b = b.permute(0, 2, 3, 1).reshape(N * H, y2, C)  # keys/vals

        q = self.q(a).view(N * H, y1, self.h, self.dk).transpose(1, 2)
        k = self.k(b).view(N * H, y2, self.h, self.dk).transpose(1, 2)
        v = self.v(b).view(N * H, y2, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / self.dk**0.5  # (.., y1, y2)
        attn = scores.softmax(dim=-1)
        out = attn @ v  # (.., y1, dk)

        out = out.transpose(1, 2).reshape(N * H, y1, C)
        out = self.proj(out)
        return out.reshape(N, H, y1, C).permute(0, 3, 1, 2)  # (N, C, H, y1)


class CardSelfAttention(nn.Module):
    """Global self-attention over the card axis of a ``(N, C, S, W)`` trunk.

    The 56-card axis ``S`` is the token sequence, each token carrying a length-``C``
    feature, attended independently for every width column ``W``. This is the
    global, content-based card-to-card mixing the conv trunk lacks -- convs only mix
    cards locally along the (bogus-locality) card axis and ``WidthCrossAttention``
    mixes *within* a card, never across cards. ``heads`` divides the feature ``C``
    (the token count ``S`` is unconstrained). Returns the same ``(N, C, S, W)``
    shape so it drops into a residual trunk.
    """

    def __init__(self, channels, heads=8):
        super().__init__()
        assert channels % heads == 0
        self.h = heads
        self.dk = channels // heads
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        # x: (N, C, S, W); tokens are the S cards, feature C, per width column W
        N, C, S, W = x.shape
        t = x.permute(0, 3, 2, 1).reshape(N * W, S, C)  # (N*W, S, C)

        q = self.q(t).view(N * W, S, self.h, self.dk).transpose(1, 2)
        k = self.k(t).view(N * W, S, self.h, self.dk).transpose(1, 2)
        v = self.v(t).view(N * W, S, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / self.dk**0.5  # (N*W, h, S, S)
        attn = scores.softmax(dim=-1)
        out = attn @ v  # (N*W, h, S, dk)

        out = out.transpose(1, 2).reshape(N * W, S, C)
        out = self.proj(out)
        return out.reshape(N, W, S, C).permute(0, 3, 2, 1)  # (N, C, S, W)
