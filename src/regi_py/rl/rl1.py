from regi_py.core import *
from regi_py.rl.utils import *
from regi_py.rl.subnets import LinearBlock, Conv1dBlock
from regi_py.strats import DamageStrategy, PreserveStrategy
import random
import time

#
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class RL1Model(torch.nn.Module):
    def __init__(self):
        super(RL1Model, self).__init__()
        self.device = "cpu"
        self.m_enemy = LinearBlock([6, 8, 16, 16])
        self.m_remy = LinearBlock([12, 16, 16])
        self.m_values = LinearBlock([1, 8, 8, 16])
        self.m_auxda = LinearBlock([6, 8, 8, 16])
        self.m_otherp = Conv1dBlock(channels=[4, 8, 8, 16], shapes=[2, 1, 1])
        self.m_combos = Conv1dBlock(channels=[8, 8, 16, 16], shapes=[2, 2, 1])
        self.m_curphd = Conv1dBlock(channels=[8, 8, 16, 16], shapes=[2, 2, 1])
        self.m_usedp = nn.RNN(
            input_size=16, hidden_size=16, batch_first=True, num_layers=2
        )
        self.nets = {
            "curphd": self.m_curphd,
            "otherp": self.m_otherp,
            "enemy": self.m_enemy,
            "combos": self.m_combos,
            "values": self.m_values,
            "usedp": self.m_usedp,
            "auxda": self.m_auxda,
            "remaining": self.m_remy,
        }

        self.bignet = Conv1dBlock(
            channels=[12, 16, 16, 16, 32, 32, 32], shapes=[1, 1, 4, 4, 2, 2]
        )
        self.make_q = nn.Sequential(
            nn.Linear(in_features=256, out_features=1), nn.ReLU()
        )

    def forward_part1(self, state, batch_size):
        res = []
        for k in self.nets.keys():
            v0 = state[k]
            if k == "usedp":
                out, v1 = self.nets[k](v0)
                v2 = v1.reshape(v1.shape[1], -1, 16)
                # print(k, v1.shape, v2.shape)
            else:
                v1 = self.nets[k](v0)
                v2 = v1.reshape(v1.shape[0], -1, 16)
                # print(k, state[k].shape, v1.shape, v2.shape)
            if v2.shape[0] != batch_size:
                v3 = v2.expand(batch_size, *v2.shape[1:])
            else:
                v3 = v2
            res.append(v3)
        return res

    def forward_part2(self, res, batch_size):
        big = torch.cat(res, dim=1)
        big2 = self.bignet(big).reshape(batch_size, -1)
        q_values = self.make_q(big2)
        return q_values

    def forward(self, state):
        batch_size = state["curphd"].shape[0]
        res = []
        for k in self.nets.keys():
            v0 = state[k]
            if k == "usedp":
                out, v1 = self.nets[k](v0)
                v2 = v1.reshape(v1.shape[1], -1, 16)
                # print(k, v1.shape, v2.shape)
            else:
                v1 = self.nets[k](v0)
                v2 = v1.reshape(v1.shape[0], -1, 16)
                # print(k, state[k].shape, v1.shape, v2.shape)
            res.append(v2)

        return self.forward_part2(res, batch_size)

    def predict(self, state):
        num_options = state["values"].shape[0]
        if num_options == 0:
            return torch.tensor([0.0]).to(self.device)
        s_common = self.tensorify([state])
        option_tens = np.zeros((num_options, 8, 4), dtype=np.float32)
        for i in range(num_options):
            option_tens[i, :] = np.matmul(
                np.diag(state["indices"][i, :]), state["curphd"]
            )
        value_tens = torch.tensor(state["values"], device=self.device)

        s_common["combos"] = torch.tensor(option_tens).to(self.device)
        s_common["values"] = value_tens.unsqueeze(1)
        res = self.forward_part1(s_common, num_options)
        return self.forward_part2(res, num_options)

    def tensorify(self, states0, batch_size=1):
        with torch.no_grad():
            return self._tensorify(states0, batch_size, self.device)

    @classmethod
    def _tensorify(cls, states0, batch_size=1, device="cpu"):
        if isinstance(states0, dict):
            states = [states0]
        else:
            states = states0

        sub = random.sample(states, batch_size)

        t_auxdata = torch.zeros((batch_size, Numberizer.AUXDA_INFO_SIZE))
        t_curphd = torch.zeros(
            (batch_size, Numberizer.MAX_HAND_SIZE, Numberizer.CARD_INFO_SIZE)
        )
        t_combos = torch.zeros(
            (batch_size, Numberizer.MAX_HAND_SIZE, Numberizer.CARD_INFO_SIZE)
        )
        t_values = torch.zeros((batch_size, 1))
        t_otherp = torch.zeros(
            (batch_size, Numberizer.MAX_PLAYERS, Numberizer.OTHERP_INFO_SIZE)
        )
        t_enemy = torch.zeros((batch_size, Numberizer.BLOCK_INFO_SIZE))
        t_remy = torch.zeros((batch_size, Numberizer.MAX_ENEMIES))
        #
        t_cur_rewards = torch.zeros((batch_size, 1))
        t_best_futures = torch.zeros((batch_size, 1))

        usedp_sizes = []
        usedp_pieces = []
        for i, s in enumerate(sub):
            t_auxdata[i, :] = torch.tensor(s["auxda"])
            t_curphd[i, :] = torch.tensor(s["curphd"])
            t_otherp[i, :] = torch.tensor(s["otherp"])
            t_enemy[i, :] = torch.tensor(s["enemy"])
            t_remy[i, :] = torch.tensor(s["remaining"])
            t_cur_rewards[i, :] = torch.tensor(s["reward"])
            t_best_futures[i, :] = torch.tensor(s["best_future"])
            #
            option = s.get("option")
            if option is not None:
                t_values[i, :] = torch.tensor(s["values"][option])
                t_combos[i, :] = torch.tensor(
                    np.matmul(np.diag(s["indices"][option, :]), s["curphd"])
                )
            #
            usedp_sizes.append(s["usedp"].shape[0])
            usedp_pieces.append(torch.tensor(s["usedp"], dtype=torch.float32))

        t_usedp_sizes = torch.tensor(usedp_sizes)
        t_usedp_padded = pad_sequence(usedp_pieces, batch_first=True, padding_value=0.0)
        t_usedp_packed = pack_padded_sequence(
            t_usedp_padded, t_usedp_sizes, batch_first=True, enforce_sorted=False
        )

        state_batch = {
            "curphd": t_curphd.to(device),
            "otherp": t_otherp.to(device),
            "enemy": t_enemy.to(device),
            "combos": t_combos.to(device),
            "values": t_values.to(device),
            "usedp": t_usedp_packed.to(device),
            "auxda": t_auxdata.to(device),
            "remaining": t_remy.to(device),
            "rewards": t_cur_rewards.to(device),
            "best_future": t_best_futures.to(device),
        }
        return state_batch


class RL1Strategy(BaseStrategy):
    __strat_name__ = "rl1"

    def __init__(self, gamma=0.9, epsilon=0.01, weights_path=None):
        super(RL1Strategy, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.numberizer = Numberizer()
        self.model = RL1Model()
        self.backup = DamageStrategy()
        if weights_path is not None:
            self.model.load_state_dict(
                torch.load(
                    weights_path, weights_only=True, map_location=self.model.device
                )
            )

    def setup(self, player, game):
        self.model.eval()
        return 0

    def getRedirectIndex(self, player, game):
        offset = random.randint(1, game.num_players - 1)
        return (game.active_player + offset) % game.num_players

    def getAttackIndex(self, combos, player, yield_allowed, game):
        state = self.numberizer.numberize_state(combos, player, game, True)
        if random.random() < self.epsilon:
            return self.backup.getAttackIndex(combos, player, yield_allowed, game)
        q_values = self.model.predict(state)
        # for i, c in enumerate(combos):
        #    print(i, c, q_values[i].item())
        option = torch.argmax(q_values).item()
        return option

    def getDefenseIndex(self, combos, player, damage, game):
        state = self.numberizer.numberize_state(combos, player, game, False)
        if random.random() < self.epsilon:
            return self.backup.getDefenseIndex(combos, player, damage, game)
        q_values = self.model.predict(state)
        # for i, c in enumerate(combos):
        #    print(i, c, q_values[i].item())
        option = torch.argmax(q_values).item()
        return option
