from regi_py.core import PhaseInfo
from regi_py.core import LocationInfo
from regi_py.strats.mcts_explorer import MCTSNodeInfo
from regi_py.rl.utils import *
from regi_py.rl.subnets import LinearBlock

#
import torch
import torch.nn as nn


class BasicNet(nn.Module):
    __mname__ = "basic"
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        #
        self.atkdef = nn.Embedding(num_embeddings=2, embedding_dim=32)
        self.net1 = LinearBlock(shapes=(9, 12))
        self.net2 = nn.Bilinear(12, 32, 16)
        self.net3 = LinearBlock(shapes=(16, 8, 4, 1))
        self.ac1 = nn.ReLU()

        self.net4a = LinearBlock(shapes=(12, 8, 4, 1))
        self.net4b = LinearBlock(shapes=(55, 8, 1))
        self.ac2 = nn.Sigmoid()

    def forward(self, data):
        y00 = self.net1(data["x"])
        v0 = self.net4a(y00).reshape(y00.shape[0], -1)
        v1 = self.ac2(self.net4b(v0))

        y01 = self.atkdef(data["attack"]).unsqueeze(1).expand(-1, y00.shape[1], -1)
        y1 = self.net2(y00, y01)
        y2 = self.net3(y1)
        y3 = self.ac1(y2.reshape(y2.shape[0], -1))
        return y3, v1

    def calculate_loss(self, y, v, y_hat, v_hat):
        loss1 = nn.functional.mse_loss(y_hat * y, y)
        loss2 = nn.functional.mse_loss(v_hat, v)
        return loss1 + loss2

    def predict(self, obj):
        info = MCTSNodeInfo(phase=str(obj), N0=0, value=0, N1=[], combos=[], sel_index=-1, offset=0)
        data = self.tensorify([info])
        y_hat0, v_hat0 = self.forward(data[0])
        y_hat = y_hat0.detach().cpu().numpy()[0]
        v_hat = v_hat0.detach().cpu().numpy()[0]
        return y_hat, v_hat

    def tensorify(self, infos):
        results = []
        for info in infos:
            phase = PhaseInfo.from_string(info.phase)
            locat = LocationInfo.from_active(phase, phase.active_player)
            x = np.array(locat, dtype=np.float32)
            #
            cards_before = set(int(l) for l in x[:, 1].nonzero()[0])
            cards_before.add(0)  # yield
            y = get_keepyness(cards_before, info)
            #
            results.append(
                {
                    # x is 1, 55, 9
                    "x": torch.from_numpy(x).unsqueeze(0),
                    # y is 1, 55
                    "y": torch.from_numpy(y).unsqueeze(0),
                    "value": torch.FloatTensor([info.value]).unsqueeze(0),
                    "attack": torch.LongTensor([phase.phase_attacking]),
                }
            )
        return results
