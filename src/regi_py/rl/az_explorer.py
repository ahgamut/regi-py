from regi_py.core import PhaseInfo
from regi_py.core import LocationInfo
from regi_py.core import ComboTable
from regi_py.core import Combo
from regi_py.strats.mcts_explorer import MCTSNode
from regi_py.rl.utils import *

#
import random
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any


@dataclass(slots=True)
class AZNodeInfo:
    history: Tuple[PhaseInfo]
    value: float
    atk_probs: Any
    keepyness: Any


class AlphaZeroNode(MCTSNode):
    rng = np.random.default_rng()
    alpha = 0.8

    def __init__(
        self,
        root_phase,
        history,
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
            parent=parent,
            prev_combo=prev_combo,
            prev_index=prev_index,
            weight=weight,
        )
        self.net = net
        self.prior = prior
        self.history = AlphaZeroNode._trimmed_history(
            history, self.root_phase, self.net.max_history
        )
        #
        v_hat, k_hat, a_hat = self.net.predict(self.history)
        self.value = v_hat
        self.keepyness = k_hat
        self.atk_probs = a_hat
        self.next_priors = np.zeros(len(self.next_combos), dtype=np.float32)
        self.atk_map = dict()

        if len(self.next_combos) != 0:
            if self.root_phase.phase_attacking:
                self._load_atk_priors()
            else:
                self._load_def_priors()
            self.next_priors = normalize_probs(1e-3 + self.next_priors)

    @staticmethod
    def _trimmed_history(history, phase, maxhist):
        tmp = history + [phase]
        if len(tmp) >= maxhist:
            return tmp[-maxhist:]
        else:
            n = len(tmp)
            return [tmp[0]] * (maxhist - n) + tmp

    def _load_def_priors(self):
        for i, combo in enumerate(self.next_combos):
            wt = sum(self.keepyness[card.location] for card in combo.parts)
            self.next_priors[i] = max(0, 1 - wt)

    def _load_atk_priors(self):
        roundabout = dict()
        N = len(self.next_combos)
        for i, combo in enumerate(self.next_combos):
            roundabout[str(sorted(combo.parts))] = i
        #
        avail = ComboTable.empty()
        avail.add_used_pile(self.next_combos)
        loc, pst = np.array(avail).nonzero()
        #
        for i in range(N):
            combo = ComboTable.make_combo(loc[i], pst[i])
            k = str(sorted(combo.parts))
            if k not in roundabout:
                print(k, roundabout)
                continue
            self.next_priors[roundabout[k]] = self.atk_probs[loc[i], pst[i]]
            self.atk_map[k] = loc[i], pst[i]
        #
        noise = self.rng.dirichlet([0.35] * N)
        self.next_priors = 0.8 * self.next_priors + 0.2 * noise

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
        new_node = AlphaZeroNode(
            phase,
            history=self.history,
            net=self.net,
            prior=prior,
            trim=self.trim,
            parent=self,
            prev_combo=combo,
            prev_index=i,
            weight=self.weight,
        )
        self.children.append(new_node)
        self.childmap[str(sorted(combo.parts))] = new_node
        return new_node

    def simulate(self):
        end_value = self.root_phase.game_endvalue
        if end_value == 1:
            return 3.0
        if end_value == -1:
            e = enemy_hp_left(self.root_phase)
            if e > 280:
                return -1
            if e > 220:
                return -0.75
            if e > 160:
                return -0.25
            return -0.0625
        return self.value

    def export(self):
        N0 = self.visits
        atk_probs = np.zeros((MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS), dtype=np.float32)
        keepyness = np.zeros(MAX_CARDS_IN_GAME, dtype=np.float32)
        #
        for card in self.root_phase.player_cards[self.root_phase.active_player]:
            keepyness[card.location] = N0
        #
        for combo in self.next_combos:
            c0 = str(sorted(combo.parts))

            if c0 in self.childmap:
                N1 = self.childmap[c0].visits
            else:
                N1 = 0

            if c0 in self.atk_map:
                loc, pst = self.atk_map[c0]
                atk_probs[loc, pst] = N1

            for card in combo.parts:
                keepyness[card.location] -= N1

        keepyness = np.maximum(0, keepyness)
        atk_probs /= N0

        return AZNodeInfo(
            history=self.history,
            value=self.value / self.visits,
            keepyness=keepyness,
            atk_probs=atk_probs,
        )


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
        root_phase = game.export_phaseinfo()
        history = AlphaZeroNode._trimmed_history(
            game.history, root_phase, self.net.max_history
        )
        vals = [-100] * game.num_players
        # TODO: how to randomize?
        for i in range(game.num_players):
            if i == game.active_player:
                continue
            root_phase._randomize()
            history[-1] = PhaseInfo.randomize_from(root_phase, i)
            v_hat, _, _ = self.net.predict(history, i)
            vals[i] = v_hat
        ind = int(np.argmax(v_hat))
        return ind

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        root_phase = game.export_phaseinfo()
        history = AlphaZeroNode._trimmed_history(
            game.history, root_phase, self.net.max_history
        )
        v_hat, k_hat, a_hat = self.net.predict(history)
        atk_priors = np.zeros(len(combos), dtype=np.float32)
        #
        roundabout = dict()
        N = len(combos)
        for i, combo in enumerate(combos):
            roundabout[str(sorted(combo.parts))] = i
        #
        avail = ComboTable.empty()
        avail.add_used_pile(combos)
        loc, pst = np.array(avail).nonzero()
        #
        for i in range(len(loc)):
            combo = ComboTable.make_combo(loc[i], pst[i])
            k = str(sorted(combo.parts))
            if k not in roundabout:
                print(k, roundabout)
                continue
            atk_priors[roundabout[k]] = a_hat[loc[i], pst[i]]

        ind = int(np.argmax(atk_priors))
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        root_phase = game.export_phaseinfo()
        history = AlphaZeroNode._trimmed_history(
            game.history, root_phase, self.net.max_history
        )
        v_hat, k_hat, a_hat = self.net.predict(history)
        def_priors = np.zeros(len(combos), dtype=np.float32)
        for i, combo in enumerate(combos):
            wt = sum(k_hat[card.location] for card in combo.parts)
            def_priors[i] = max(0, 1 - wt)
        ind = int(np.argmax(def_priors))
        return ind
