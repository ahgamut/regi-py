from regi_py.core import PhaseInfo
from regi_py.core import LocationInfo
from regi_py.core import MAX_CARDS_IN_GAME
from regi_py.rl.az_explorer import AZNodeInfo
from regi_py.rl.utils import *
from regi_py.rl.subnets import (
    LinearBlock,
    Conv1dBlock,
    Conv2dBlock,
    WidthCrossAttention,
)

#
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = Conv2dBlock(channels=(64, 8, 1), shapes=(3, 1), paddings=(1, 0))
        self.net2 = nn.Linear(in_features=495, out_features=1)
        self.ac = nn.Sigmoid()

    def forward(self, x):
        x = self.net1(x).reshape(x.shape[0], -1)
        x = self.net2(x)
        x = self.ac(x)
        return x


class KeepyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = Conv2dBlock(channels=(64, 8, 1), shapes=(3, 1), paddings=(1, 0))
        self.net2 = nn.Linear(in_features=495, out_features=MAX_CARDS_IN_GAME)
        self.ac = nn.Tanh()

    def forward(self, x):
        x = self.net1(x).reshape(x.shape[0], -1)
        x = self.net2(x)
        x = self.ac(x)
        return x


class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        n = MAX_CARDS_IN_GAME
        self.ac = nn.Softmax2d()
        self.net1 = Conv2dBlock(channels=(64, 16, 4), shapes=(3, 1), paddings=(1, 0))
        self.net2 = Conv1dBlock(
            channels=(n, n, n, n), shapes=(7, 7, 3), paddings=(0, 0, 0)
        )
        self.wca = WidthCrossAttention(channels=MAX_CARDS_IN_GAME, heads=5)

    def forward(self, x0, k):
        x = self.net1(x0)
        x = x.reshape(x0.shape[0], MAX_CARDS_IN_GAME, -1)
        x = self.net2(x)
        x = x.reshape(x.shape[0], MAX_CARDS_IN_GAME, 1, MAX_PLAYED_STATUS)
        k2 = k.reshape(x.shape[0], MAX_CARDS_IN_GAME, 1, 1)
        x = self.wca(x, k2).reshape(-1, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)
        x = self.ac(x)
        return x


class CombineNet(nn.Module):
    def __init__(self, channels=32, reduction=4):
        super().__init__()
        self.wca = WidthCrossAttention(channels=channels, heads=4)
        self.net = Conv2dBlock(
            channels=(channels, 64, 64, 64, 64, 64),
            shapes=(1, 3, 3, 3, 3),
            paddings=(0, 1, 1, 1, 1),
        )

    def forward(self, x1, x2):
        y1 = self.wca(x1, x2)
        y2 = self.net(y1)
        return y2


class BasicNet(nn.Module):
    __mname__ = "basic"

    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.max_history = 8
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

        self.combiner = CombineNet(channels=32)
        self.v_net = ValueNet()
        self.k_net = KeepyNet()
        self.a_net = ActionNet()

    def forward(self, data):
        x1 = self.loc_net(data["location"])
        x2 = self.usp_net(data["used_pile"])
        x = self.combiner(x1, x2)
        v = self.v_net(x)
        k = self.k_net(x)
        a = self.a_net(x, k)
        return v, k, a

    def calculate_loss(self, y, y_hat, phase_atk):
        v, k, a = y
        v_hat, k_hat, a_hat = y_hat
        loss1a = torch.sum(-a * torch.log(a_hat), dim=(-2, -1))
        loss1 = torch.mean(loss1a * phase_atk)
        loss2 = nn.functional.mse_loss(v_hat, v)
        loss3 = nn.functional.mse_loss(k_hat * k, k)
        return loss1 + loss2 + loss3

    def predict(self, history, perspective=None):
        data = BasicNet.tensorify_phases(history, perspective, self.max_history)
        v_hat0, k_hat0, a_hat0 = self.forward(data)
        v_hat = float(v_hat0.detach().cpu().numpy()[0, 0])
        k_hat = k_hat0.detach().cpu().numpy()[0, :]
        a_hat = a_hat0.detach().cpu().numpy()[0, 0, :, :]
        return v_hat, k_hat, a_hat

    @staticmethod
    def tensorify_phases(history, perspective=None, window=8):
        result = {
            "location": torch.zeros((1, window, MAX_CARDS_IN_GAME, MAX_LOCATIONS)),
            "used_pile": torch.zeros((1, window, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)),
        }
        if perspective is None:
            perspective = history[-1].active_player
        #
        for j in range(window):
            phase = history[j]
            loca0 = np.array(
                LocationInfo.from_current(phase, perspective), dtype=np.float32
            )
            locat = loca0 / loca0.sum(axis=1, keepdims=True)
            table = np.array(ComboTable.from_phase(phase), dtype=np.float32)
            result["location"][0, j] = torch.from_numpy(locat)
            result["used_pile"][0, j] = torch.from_numpy(table)
        return result

    @staticmethod
    def tensorify_training(infos):
        N = len(infos)
        window = len(infos[0].history)
        result = {
            "location": torch.zeros((N, window, MAX_CARDS_IN_GAME, MAX_LOCATIONS)),
            "used_pile": torch.zeros((N, window, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)),
            "value": torch.zeros((N, 1)),
            "keepyness": torch.ones((N, MAX_CARDS_IN_GAME)),
            "atk_probs": torch.zeros((N, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)),
            "attacking": torch.zeros((N, 1)),
        }
        #
        for i in range(N):
            info = infos[i]
            cur_phase = info.history[-1]
            result["value"][i, 0] = info.value
            result["attacking"][i, 0] = cur_phase.phase_attacking
            result["keepyness"][i, :] = torch.from_numpy(info.keepyness)
            result["atk_probs"][i, 0] = torch.from_numpy(info.atk_probs)
            #
            perspective = cur_phase.active_player
            for j in range(window, 0, -1):
                phase = info.history[-j]
                loca0 = np.array(
                    LocationInfo.from_current(phase, perspective), dtype=np.float32
                )
                locat = loca0 / loca0.sum(axis=1, keepdims=True)
                table = np.array(ComboTable.from_phase(phase), dtype=np.float32)
                result["location"][i, -j] = torch.from_numpy(locat)
                result["used_pile"][i, -j] = torch.from_numpy(table)

        return result
