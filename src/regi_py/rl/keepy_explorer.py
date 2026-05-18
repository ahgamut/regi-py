from regi_py.core import PhaseInfo
from regi_py.core import LocationInfo
from regi_py.strats.mcts_explorer import MCTSNode
from regi_py.rl.utils import *

#
import random
import math
import numpy as np


class KeepyPUCTNode(MCTSNode):
    def __init__(
        self,
        root_phase,
        net,
        prior,
        trim=False,
        parent=None,
        prev_combo=None,
        prev_index=None,
        weight=math.sqrt(2),
    ):
        super().__init__(
            root_phase=root_phase,
            trim=trim,
            parent=None,
            prev_combo=prev_combo,
            prev_index=prev_index,
            weight=weight,
        )
        self.net = net
        self.prior = prior

        #
        y_hat, v_hat = self.net.predict(root_phase)
        self.value += v_hat
        preds = 1 / (1.0 + y_hat)
        self.next_priors = np.zeros(len(self.next_combos), dtype=np.float32)
        for i, combo in enumerate(self.next_combos):
            wt = sum(preds[card.location] for card in combo.parts)
            self.next_priors[i] = wt
        self.next_priors /= (1.0 + np.sum(self.next_priors))

    @property
    def ucb1(self):
        if self.visits == 0:
            return float("inf")
        v1 = self.value / self.visits
        if self.parent:
            v2 = math.sqrt(self.parent.visits) / (1 + self.visits)
        else:
            v2 = 0
        return v1 + self.prior * self.weight * v2

    def expand(self):
        i = self.rem_exp_ind.pop()
        phase = self.next_phases[i]
        combo = self.next_combos[i]
        prior = self.next_priors[i]
        new_node = KeepyPUCTNode(
            phase,
            net=self.net,
            prior=prior,
            trim=self.trim,
            parent=self,
            prev_combo=combo,
            prev_index=i,
            weight=self.weight,
        )
        self.children.append(new_node)
        self.childmap[str(combo)] = new_node
        return new_node


class PUCTExplorerStrategy(BaseStrategy):
    __strat_name__ = "puct-explorer"

    def __init__(self, net, iterations=64, trim=True, weight=math.sqrt(2)):
        super(PUCTExplorerStrategy, self).__init__()
        self.net = net
        self.iterations = iterations
        self.__strat_name__ = f"puct-{net.__mname__}-{iterations}"
        self.trim = trim
        self.weight = weight

    def setup(self, player, game):
        self.net.eval()
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
        root_node = KeepyPUCTNode(
            phase, net=self.net, prior=1.0, trim=self.trim, weight=self.weight
        )

        for i in range(self.iterations):
            node = MCTSNode.select(root_node)
            if not node.is_terminal():
                node = node.expand()
            reward = node.simulate()
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


class NetDirectStrategy(BaseStrategy):
    __strat_name__ = "net-direct"

    def __init__(self, net):
        super(NetDirectStrategy, self).__init__()
        self.net = net
        self.__strat_name__ = f"direct-{net.__mname__}"

    def setup(self, player, game):
        self.net.eval()
        return 0

    def getRedirectIndex(self, player, game):
        offset = random.randint(1, game.num_players - 1)
        next_player = (game.active_player + offset) % game.num_players
        return next_player

    def process_phase(self, phase, combos):
        y_hat, v_hat = self.net.predict(phase)
        throw = 0 - y_hat
        best_ind = 0
        best_score = 100
        for ind, combo in enumerate(combos):
            if len(combo.parts) == 0:
                score = throw[0]
            else:
                score = sum(throw[card.location] for card in combo.parts)
            if score < best_score:
                best_ind = ind
                best_score = score
        return ind

    def getAttackIndex(self, combos, player, yield_allowed, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        return ind
