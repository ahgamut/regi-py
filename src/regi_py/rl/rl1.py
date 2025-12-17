from regi_py.core import *
from regi_py.rl.utils import *
from regi_py.rl.subnets import LinearBlock, Conv1dBlock
import random

#
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class RL1Model(torch.nn.Module):
    def __init__(self):
        super(RL1Model, self).__init__()
        self.m_enemy = LinearBlock([6, 8, 16, 16])
        self.m_remy = LinearBlock([12, 16, 16])
        self.m_values = LinearBlock([1, 8, 8, 16])
        self.m_auxda = LinearBlock([6, 8, 8, 16])
        self.m_otherp = Conv1dBlock(channels=[4, 8, 8, 16], shapes=[2, 1, 1])
        self.m_combos = Conv1dBlock(channels=[8, 8, 16, 16], shapes=[2, 2, 1])
        self.m_curphd = Conv1dBlock(channels=[8, 8, 16, 16], shapes=[2, 2, 1])
        self.m_usedp = nn.RNN(input_size=16, hidden_size=16, batch_first=True, num_layers=2)
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

        self.bignet = Conv1dBlock(channels=[12, 16, 16, 16, 32, 32, 32], shapes=[1, 1, 4, 4, 2, 2])
        self.make_q = nn.Sequential(nn.Linear(in_features=256, out_features=1), nn.ReLU())

    def forward(self, state):
        batch_size = state["curphd"].shape[0]
        res = []
        for k in self.nets.keys():
            if k == "usedp":
                out, v1 = self.nets[k](state[k])
                v2 = v1.reshape(v1.shape[1], -1, 16)
                # print(k, v1.shape, v2.shape)
            else:
                v1 = self.nets[k](state[k])
                v2 = v1.reshape(v1.shape[0], -1, 16)
                # print(k, state[k].shape, v1.shape, v2.shape)
            res.append(v2)

        big = torch.cat(res, dim=1)
        big2 = self.bignet(big).reshape(batch_size, -1)
        q_values = self.make_q(big2)
        # print("q_values", q_values.shape)
        # print(state)
        return q_values

    def predict(self, state):
        num_options = state["values"].shape[0]
        if num_options == 0:
            return torch.Tensor([0.0])
        q_values = torch.zeros(num_options)
        for i in range(num_options):
            q_values[i] = self.rate_combo(state, i)[0]
        return q_values

    def rate_combo(self, state, ind):
        s2 = [dict(**state)]
        s2[0]["option"] = ind
        s2[0]["reward"] = state.get("reward", 0)
        s2[0]["best_future"] = state.get("best_future", 0)
        s2[0]["best_from_here"] = state.get("best_from_here", 0)
        tens = self.tensorify(s2)
        return self.forward(tens)

    @classmethod
    def tensorify(cls, states0, batch_size=1):
        with torch.no_grad():
            return cls._tensorify(states0, batch_size)

    @classmethod
    def _tensorify(cls, states0, batch_size=1):
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
            option = s["option"]
            t_values[i, :] = torch.tensor(s["values"][option])
            t_combos[i, :] = torch.tensor(np.matmul(s["indices"][option], s["curphd"]))
            #
            usedp_sizes.append(s["usedp"].shape[0])
            usedp_pieces.append(torch.tensor(s["usedp"], dtype=torch.float32))

        t_usedp_sizes = torch.tensor(usedp_sizes)
        t_usedp_padded = pad_sequence(usedp_pieces, batch_first=True, padding_value=0.0)
        t_usedp_packed = pack_padded_sequence(
            t_usedp_padded, t_usedp_sizes, batch_first=True, enforce_sorted=False
        )

        state_batch = {
            "curphd": t_curphd,
            "otherp": t_otherp,
            "enemy": t_enemy,
            "combos": t_combos,
            "values": t_values,
            "usedp": t_usedp_packed,
            "auxda": t_auxdata,
            "remaining": t_remy,
            "rewards": t_cur_rewards,
            "best_future": t_best_futures,
        }
        return state_batch


class RL1Strategy(BaseStrategy):
    __strat_name__ = "rl1"

    def __init__(self, gamma=0.9, epsilon=1.0, weights_path=None):
        super(RL1Strategy, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.numberizer = Numberizer()
        self.model = RL1Model()
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))

    def setup(self, player, game):
        self.model.eval()
        return 0

    def select_action(self, combos, player, game, is_attacking):
        if len(combos) == 0:
            return -1
        state = self.numberizer.numberize_state(combos, player, game, is_attacking)
        if random.random() < self.epsilon:
            return random.randint(0, len(combos) - 1)
        q_values = self.model.predict(state)
        option = torch.argmax(q_values).item()
        return option

    def getAttackIndex(self, combos, player, yield_allowed, game):
        return self.select_action(combos, player, game, True)

    def getDefenseIndex(self, combos, player, damage, game):
        return self.select_action(combos, player, game, False)
