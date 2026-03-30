from regi_py.core import BaseStrategy
from regi_py.core import RandomStrategy
from regi_py.strats.recommender import RecommenderMixin
from regi_py.strats.phase_utils import *
from regi_py.strats.sub_random import SubsetRandomStrategy

#
import random
import math
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(slots=True)
class MCTSNodeInfo:
    phase: str
    value: float
    N0: int  # visits of this node
    N1: Tuple[int]  # uses of combos from this node
    combos: Tuple[str]
    sel_index: int  # index of combo selected at this node
    offset: int  # nonzero if a redirect happened, the offset from the current player


class MCTSNode:
    def __init__(
        self,
        root_phase,
        trim=False,
        parent=None,
        prev_combo=None,
        prev_index=None,
        weight=math.sqrt(2),
    ):
        self.root_phase = root_phase
        self.trim = trim
        self.parent = parent
        self.weight = weight
        self.prev_combo = prev_combo
        self.prev_index = prev_index
        #
        self.next_phases = []
        self.next_combos = []
        self.rem_exp_ind = []
        self.children = []
        self.childmap = dict()
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

    def export(self):
        combos = []
        policy = []
        for combo in self.next_combos:
            c0 = str(combo)
            combos.append(c0)
            if c0 in self.childmap:
                policy.append(self.childmap[c0].visits)
            else:
                policy.append(0)

        if len(self.children) > 0:
            sel_index = self.best_child_node.prev_index
        else:
            sel_index = 0

        return MCTSNodeInfo(
            phase=str(self.root_phase),
            value=self.value,
            N0=self.visits,
            N1=tuple(policy),
            combos=tuple(combos),
            sel_index=sel_index,
            offset=0,  # zero means no redirect happened from this move
        )

    @property
    def best_child_node(self):
        ind = 0
        mvx = self.children[0].visits
        for i, x in enumerate(self.children):
            if x.visits > mvx:
                ind = i
                mvx = x.visits
        return self.children[ind]

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
            phase,
            trim=node.trim,
            parent=node,
            prev_combo=combo,
            prev_index=i,
            weight=node.weight,
        )
        node.children.append(new_node)
        node.childmap[str(combo)] = new_node
        return new_node

    @staticmethod
    def simulate(node):
        end_value = node.root_phase.game_endvalue
        if end_value != 0:
            return float(end_value == 1)

        end_game, _ = quick_game_sim(node.root_phase, strat_klass=SubsetRandomStrategy)
        s = enemy_hp_left(node.root_phase)
        e = enemy_hp_left(end_game)
        end_value = (360 - e) / 360
        pacing = (s - e) / end_game.phase_count
        reward = end_game.phase_count / 50
        # penalize games that are immediate losses (throwy)
        if end_game.phase_count <= 3 and e > 0:
            if s > 280:
                return -1
            if s > 220:
                return -0.75
            if s > 160:
                return -0.25
            return -0.0625
        # more if checkpoints are crossed
        if s > 280 and e <= 220:
            reward += end_value
        if s > 220 and e <= 160:
            reward += end_value
        if s > 160 and e <= 120:
            reward += end_value
        if s > 120 and e <= 80:
            reward += end_value
        if s > 80 and e <= 40:
            reward += end_value
        if s > 40 and e <= 0:
            reward += 3 * end_value
        # penalize games that are too slow-paced
        if pacing < 2.1:
            return reward / 2
        return reward

    @staticmethod
    def update(node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class MCTSExplorerStrategy(BaseStrategy, RecommenderMixin):
    __strat_name__ = "mcts-explorer"

    def __init__(self, iterations=64, trim=True, weight=math.sqrt(2), num_recos=5):
        super(MCTSExplorerStrategy, self).__init__()
        self.iterations = iterations
        self.__strat_name__ = f"mcts-{iterations}"
        self.trim = trim
        self.weight = weight
        self.num_recos = num_recos

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
        # for i, x in enumerate(root_node.children):
        #    print(x.prev_combo, x.visits / root_node.visits)
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

    def getRecommendedMoves(self, phase, combos):
        root_node = self.simulate_node(phase)
        info = root_node.export()
        moves = info.combos
        scores = info.N1
        sinds = np.argsort(scores)[::-1]
        nr = min(self.num_recos, len(scores))
        recos = [moves[int(x)] for x in sinds[:nr]]
        return recos


class MCTSSaverStrategy(MCTSExplorerStrategy):
    __strat_name__ = "mcts-saver"

    def __init__(self, iterations=64, trim=True, weight=math.sqrt(2)):
        super(MCTSSaverStrategy, self).__init__(iterations, trim, weight)
        self.history = []

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
            # redirect because previous attack was JOKER
            self.history[-1].offset = offset
            next_player = (game.active_player + offset) % game.num_players
        return next_player

    def process_phase(self, phase, combos):
        root_node = self.simulate_node(phase)
        best_combo = root_node.best_combo
        info = root_node.export()
        info.sel_index = root_node.best_child_node.prev_index
        self.history.append(info)
        # for i, x in enumerate(root_node.children):
        #    print(x.prev_combo, x.visits / root_node.visits)
        for ind, c in enumerate(combos):
            if c.bitwise == best_combo.bitwise:
                return ind
        return -1
