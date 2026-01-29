import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, shapes):
        super(LinearBlock, self).__init__()
        self.ac = nn.LeakyReLU(0.05)
        self.nets = []
        for i in range(0, len(shapes) - 1):
            self.nets.append(
                nn.Linear(in_features=shapes[i], out_features=shapes[i + 1])
            )

    def forward(self, x0):
        x = x0
        for i, net in enumerate(self.nets):
            y = self.ac(net(x))
            if x.shape == y.shape:
                x = x + y
            else:
                x = y
        return x


class Conv1dBlock(nn.Module):
    def __init__(self, shapes, channels, paddings):
        super(Conv1dBlock, self).__init__()
        self.ac = nn.LeakyReLU(0.05)
        self.nets = []
        self.feeds = []
        for i in range(0, len(channels) - 1):
            self.nets.append(
                nn.Sequential(
                    nn.Conv1d(
                        kernel_size=shapes[i],
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        padding=paddings[i],
                        bias=True,
                    ),
                    nn.BatchNorm1d(num_features=channels[i + 1]),
                )
            )

    def forward(self, x0):
        x = x0
        for i, net in enumerate(self.nets):
            y = self.ac(net(x))
            if y.shape == x.shape:
                x = x + y
            else:
                x = y
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, shapes, channels, paddings):
        super(Conv2dBlock, self).__init__()
        self.ac = nn.LeakyReLU(0.05)
        self.nets = []
        self.feeds = []
        for i in range(0, len(channels) - 1):
            self.nets.append(
                nn.Sequential(
                    nn.Conv2d(
                        kernel_size=shapes[i],
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        padding=paddings[i],
                    ),
                    nn.BatchNorm2d(num_features=channels[i + 1]),
                )
            )

    def forward(self, x0):
        x = x0
        for i, net in enumerate(self.nets):
            y = self.ac(net(x))
            if y.shape == x.shape:
                x = x + y
            else:
                x = y
        return x
