from regi_py.core import BaseStrategy
from regi_py.core import RandomStrategy
from regi_py.strats.phase_utils import *
from regi_py.strats.trim_random import TrimmedRandomStrategy
import random
import math


class MCTSNode:
    def __init__(
        self, root_phase, trim=False, parent=None, prev_combo=None, weight=math.sqrt(2)
    ):
        self.root_phase = root_phase
        self.trim = trim
        self.parent = parent
        self.weight = weight
        self.prev_combo = prev_combo
        #
        self.next_phases = []
        self.next_combos = []
        self.rem_exp_ind = []
        self.children = []
        self.visits = 0
        self.value = 0.0
        self._load_expansion()

    def _load_expansion(self):
        if self.root_phase.game_endvalue != 0:
            return
        n, c = get_expansion_at(self.root_phase, trim=self.trim)
        self.next_phases = n
        self.next_combos = c
        assert len(n) == len(c)
        self.rem_exp_ind = list(range(len(c)))
        random.shuffle(self.rem_exp_ind)

    def can_expand_further(self):
        return len(self.rem_exp_ind) != 0

    def is_terminal(self):
        return self.root_phase.game_endvalue != 0

    @property
    def best_child_node(self):
        return max(self.children, key=lambda n: n.visits)

    @property
    def best_combo(self):
        return self.best_child_node.prev_combo

    @property
    def best_next_phase(self):
        return self.best_child_node.root_phase

    @property
    def ucb1(self):
        if self.visits == 0:
            return float("inf")
        v1 = self.value / self.visits
        v2 = math.sqrt(math.log(self.parent.visits) / self.visits)
        return v1 + self.weight * v2

    @staticmethod
    def select(node):
        while not node.can_expand_further():
            if node.is_terminal():
                break
            node = max(node.children, key=lambda n: n.ucb1)
        return node

    @staticmethod
    def expand(node):
        i = node.rem_exp_ind.pop()
        phase = node.next_phases[i]
        combo = node.next_combos[i]
        new_node = MCTSNode(
            phase, trim=node.trim, parent=node, prev_combo=combo, weight=node.weight
        )
        node.children.append(new_node)
        return new_node

    @staticmethod
    def simulate(node):
        end_value = node.root_phase.game_endvalue
        if end_value != 0:
            return float(end_value == 1)

        end_value = quick_game_value(
            node.root_phase, strat_klass=TrimmedRandomStrategy, relative_diff=True
        )
        return end_value

    @staticmethod
    def update(node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class MCTSExplorerStrategy(BaseStrategy):
    __strat_name__ = "mcts-explorer"

    def __init__(self, iterations=1024, trim=True, weight=math.sqrt(2)):
        super(MCTSExplorerStrategy, self).__init__()
        self.iterations = iterations
        self.trim = trim
        self.weight = weight

    def setup(self, player, game):
        return 0

    def getRedirectIndex(self, player, game):
        root_node = self.simulate_node(game.export_phaseinfo())
        best_phase = root_node.best_next_phase
        ct = -1
        for i, p in enumerate(root_node.children):
            if p.visits > ct and p.root_phase.active_player != player.id:
                ct = p.visits
                best_phase = p.root_phase
        if best_phase.active_player != player.id:
            next_player = best_phase.active_player
        else:
            offset = random.randint(1, game.num_players - 1)
            next_player = (game.active_player + offset) % game.num_players
        return next_player

    def simulate_node(self, phase):
        root_node = MCTSNode(phase, trim=self.trim, weight=self.weight)

        for i in range(self.iterations):
            node = MCTSNode.select(root_node)
            if not node.is_terminal():
                node = MCTSNode.expand(node)
            reward = MCTSNode.simulate(node)
            MCTSNode.update(node, reward)
        return root_node

    def process_phase(self, phase, combos):
        root_node = self.simulate_node(phase)
        best_combo = root_node.best_combo
        for ind, c in enumerate(combos):
            if c.bitwise == best_combo.bitwise:
                return ind
        return -1

    def getAttackIndex(self, combos, player, yield_allowed, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        return ind
