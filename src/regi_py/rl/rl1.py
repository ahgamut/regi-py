from regi_py.core import *
from regi_py.rl.utils import *
import random

#
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class RL1Model(torch.nn.Module):
    def __init__(self):
        super(RL1Model, self).__init__()

    def forward(self, state):
        return torch.Tensor([0.0])

    def predict(self, state):
        print(state["status"], state["values"].shape)
        return torch.Tensor([0.0])

    def rate_attacks(self, state):
        pass

    def rate_defends(self, state):
        pass

    @classmethod
    def tensorify(cls, states0, batch_size=1):
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
            print(s["curphd"].shape, s["indices"].shape)
            t_combos[i, :] = torch.tensor(np.matmul(s["curphd"], s["indices"][option]))
            #
            usedp_sizes.append(len(s["usedp"]))
            usedp_pieces.append(torch.tensor(s["usedp"]))

        t_usedp_sizes = torch.tensor(usedp_sizes)
        t_usedp_padded = pad_sequence(usedp_pieces, batch_first=True, padding_value=0.0)
        t_usedp_packed = pack_padded_sequence(
            t_usedp_packed, t_usedp_sizes, batch_first=True, enforce_sorted=False
        )

        state_batch = {
            "curphd": t_curphd,
            "otherp": t_otherp,
            "enemy": t_enemy,
            "combos": t_combos,
            "values": t_values,
            "usedp": t_usedp_packed,
            "auxda": t_auxda,
            "remaining": t_remy,
            "rewards": t_cur_rewards,
            "best_futures": t_best_futures,
        }
        return state_batch


class RL1Strategy(BaseStrategy):
    __strat_name__ = "rl1"

    def __init__(self, gamma=0.9, eps_max=1.0, eps_min=0.1, weights_path=None):
        super(RL1Strategy, self).__init__()
        self.gamma = gamma
        self.eps_max = eps_max
        self.eps_min = eps_min
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
        if True or random.random() > self.epsilon:
            return random.randint(0, len(combos) - 1)
        q_values = self.model.predict(state)
        option = torch.argmax(q_values).item()
        return random.randint(0, len(combos) - 1)

    def getAttackIndex(self, combos, player, yield_allowed, game):
        return self.select_action(combos, player, game, True)

    def getDefenseIndex(self, combos, player, damage, game):
        return self.select_action(combos, player, game, False)
