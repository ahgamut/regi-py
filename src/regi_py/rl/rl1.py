from regi_py.core import *
from regi_py.rl.utils import *
import random
import torch


class RL1Model(torch.nn.Module):
    def __init__(self):
        super(RL1Model, self).__init__()

    def forward(self, state):
        return torch.Tensor([0.0])

    def predict(self, state):
        return torch.Tensor([0.0])

    def rate_attacks(self, state):
        pass

    def rate_defends(self, state):
        pass


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
