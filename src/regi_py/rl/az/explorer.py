from regi_py.core import PhaseInfo, BaseStrategy
from regi_py.combomap import cell_of_bitwise
from regi_py.strats.mcts_explorer import MCTSNode
from regi_py.strats.recommender import RecommenderMixin
from regi_py.strats.phase_utils import index_of_bitwise
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
        self.atk_map = dict()
        #
        # a terminal leaf has no children and its backup value comes from
        # simulate()'s endgame reward, so the net eval is never read -- skip it
        if self.root_phase.game_endvalue != 0:
            self.leaf_value = 0.0
            self.keepyness = None
            self.atk_probs = None
            self.next_priors = np.zeros(0, dtype=np.float32)
            return
        #
        v_hat, k_hat, a_hat = self.net.predict(self.history)
        # net value estimate for this leaf; kept separate from ``self.value``,
        # which the base MCTSNode uses as the backed-up reward sum (W). Folding
        # v_hat into self.value would double-count it: __init__ would seed W with
        # v_hat and then update() would add it again on the leaf's own backup.
        self.leaf_value = v_hat
        self.keepyness = k_hat
        self.atk_probs = a_hat
        self.next_priors = np.zeros(len(self.next_combos), dtype=np.float32)

        if len(self.next_combos) != 0:
            if self.root_phase.phase_attacking:
                self._load_atk_priors()
            else:
                self._load_def_priors()
            self.next_priors = normalize_probs(1e-3 + self.next_priors)
            # order expansion by prior once, ascending, so expand() pop()s the
            # highest-prior child first in O(1) (no per-expand scan). Stable sort:
            # equal-prior children keep the base class's random shuffle order.
            self.rem_exp_ind.sort(key=lambda j: self.next_priors[j])

    @staticmethod
    def _trimmed_history(history, phase, maxhist):
        tmp = list(history) + [phase]
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
        # each combo's ComboTable cell comes straight from its canonical bitwise
        # identity (combomap); read the net's per-cell prior into combo order
        for i, combo in enumerate(self.next_combos):
            lp = cell_of_bitwise(combo.bitwise)
            if lp is None:
                continue
            self.next_priors[i] = self.atk_probs[lp]
            self.atk_map[combo.bitwise] = lp

    def add_dirichlet_noise(self, alpha=0.35, frac=0.2):
        """Mix Dirichlet exploration noise into this node's child priors.

        AlphaZero adds this only at the *search root*, so it lives here as an
        explicit call (self-play drivers apply it per move) rather than in every
        node's ``__init__`` -- internal nodes should search on the raw net priors,
        and competitive play (``AZExplorerStrategy``) skips it entirely. No-op for
        defense/terminal nodes (only attack nodes carry net atk priors).
        """
        n = len(self.next_combos)
        if n == 0 or not self.root_phase.phase_attacking:
            return
        noise = self.rng.dirichlet([alpha] * n)
        self.next_priors = normalize_probs(
            (1 - frac) * self.next_priors + frac * noise
        )

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
        # rem_exp_ind is pre-sorted ascending by prior in __init__, so pop() takes
        # the highest-prior unexpanded child first. This makes the net policy -- not
        # the base MCTSNode's random shuffle -- order exploration, which is what
        # makes the PUCT prior steer search at the many deep nodes that never fully
        # expand within the budget (ucb1's PUCT term only governs already-expanded
        # children). O(1) per expand, no per-node scan.
        i = self.rem_exp_ind.pop()
        combo = self.next_combos[i]
        prior = self.next_priors[i]
        phase = self._expander.step(combo.bitwise)
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
        self.childmap[combo.bitwise] = new_node
        if not self.rem_exp_ind:
            self._expander = None  # fully expanded: release the throwaway game
        return new_node

    def simulate(self):
        end_value = self.root_phase.game_endvalue
        # keep backups on the same [-1, 1] scale as the tanh value head and the
        # z training target: win -> 1.0, loss -> shaped negative, else net estimate
        if end_value == 1:
            return 1.0
        if end_value == -1:
            return hp_loss_penalty(enemy_hp_left(self.root_phase))
        return self.leaf_value

    def export(self):
        N0 = self.visits
        atk_probs = np.zeros((MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS), dtype=np.float32)
        keepyness = np.zeros(MAX_CARDS_IN_GAME, dtype=np.float32)
        #
        for card in self.root_phase.player_cards[self.root_phase.active_player]:
            keepyness[card.location] = N0
        #
        for combo in self.next_combos:
            c0 = combo.bitwise

            if c0 in self.childmap:
                N1 = self.childmap[c0].visits
            else:
                N1 = 0

            if c0 in self.atk_map:
                loc, pst = self.atk_map[c0]
                atk_probs[loc, pst] = N1

            for card in combo.parts:
                keepyness[card.location] -= N1

        # normalize both to visit fractions in [0, 1] (N0 = root visits), matching
        # the sigmoid keepy head and the softmax action target
        keepyness = np.maximum(0, keepyness) / N0
        atk_probs /= N0

        return AZNodeInfo(
            history=self.history,
            # placeholder: the self-play driver overwrites every info.value with
            # the whole-game outcome z, so the search Q here is never used
            value=0.0,
            keepyness=keepyness,
            atk_probs=atk_probs,
        )


def simulate_node(root_node, iterations):
    """Run ``iterations`` of net-guided MCTS from ``root_node`` (in place)."""
    for _ in range(iterations):
        node = AlphaZeroNode.select(root_node)
        if not node.is_terminal():
            node = node.expand()
        reward = node.simulate()
        AlphaZeroNode.update(node, reward)
    return root_node


class NetDirectStrategy(BaseStrategy, RecommenderMixin):
    __strat_name__ = "net-direct"
    num_recos = 5

    def __init__(self, net):
        super(NetDirectStrategy, self).__init__()
        self.net = net
        self.__strat_name__ = f"direct-{net.__mname__}"

    def setup(self, player, game):
        self.net.eval()
        return 0

    def getRecommendedMoves(self, phase, combos):
        # rank the offered combos by the net's own priors (no search): attack
        # phases use the atk-cell priors, defense phases fall back to the
        # keepyness-derived score (the AZ net has no defense policy head). Returns
        # Combo objects, the unified recommender contract.
        if len(combos) == 0:
            return []
        history = AlphaZeroNode._trimmed_history([], phase, self.net.max_history)
        _, k_hat, a_hat = self.net.predict(history)
        scores = np.zeros(len(combos), dtype=np.float32)
        if phase.phase_attacking:
            for i, combo in enumerate(combos):
                lp = cell_of_bitwise(combo.bitwise)
                if lp is not None:
                    scores[i] = a_hat[lp]
        else:
            for i, combo in enumerate(combos):
                wt = sum(k_hat[card.location] for card in combo.parts)
                scores[i] = max(0.0, 1 - wt)
        order = np.argsort(scores)[::-1][: self.num_recos]
        return [combos[int(i)] for i in order]

    def getRedirectIndex(self, player, game):
        root_phase = game.export_phaseinfo()
        history = AlphaZeroNode._trimmed_history(
            game.history, root_phase, self.net.max_history
        )
        vals = [-100.0] * game.num_players
        for i in range(game.num_players):
            if i == game.active_player:
                continue
            history[-1] = PhaseInfo.randomize_from(root_phase, i)
            v_hat, _, _ = self.net.predict(history, i)
            vals[i] = v_hat
        return int(np.argmax(vals))

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        root_phase = game.export_phaseinfo()
        history = AlphaZeroNode._trimmed_history(
            game.history, root_phase, self.net.max_history
        )
        v_hat, k_hat, a_hat = self.net.predict(history)
        atk_priors = np.zeros(len(combos), dtype=np.float32)
        for i, combo in enumerate(combos):
            lp = cell_of_bitwise(combo.bitwise)
            if lp is not None:
                atk_priors[i] = a_hat[lp]
        return int(np.argmax(atk_priors))

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
        return int(np.argmax(def_priors))


class AZExplorerStrategy(BaseStrategy, RecommenderMixin):
    """Net-guided MCTS as a playable strategy (the AlphaZero analogue of
    ``strats.mcts_explorer.MCTSExplorerStrategy``): run search from the current
    state and play the most-visited child."""

    __strat_name__ = "az-explorer"

    def __init__(self, net, iterations=64, trim=True, weight=math.sqrt(2)):
        super(AZExplorerStrategy, self).__init__()
        self.net = net
        self.iterations = iterations
        self.trim = trim
        self.weight = weight
        self.__strat_name__ = f"az-{net.__mname__}-{iterations}"

    def setup(self, player, game):
        self.net.eval()
        return 0

    def _root_from_game(self, game):
        return AlphaZeroNode(
            game.export_phaseinfo(),
            history=list(game.history),
            net=self.net,
            prior=1.0,
            trim=self.trim,
            weight=self.weight,
        )

    def _search_index(self, game, combos):
        if len(combos) == 0:
            return -1
        root = simulate_node(self._root_from_game(game), self.iterations)
        if len(root.children) == 0:
            return -1
        return index_of_bitwise(combos, root.best_combo.bitwise)

    def getAttackIndex(self, combos, player, yield_allowed, game):
        return self._search_index(game, combos)

    def getDefenseIndex(self, combos, player, damage, game):
        return self._search_index(game, combos)

    def getRedirectIndex(self, player, game):
        root = simulate_node(self._root_from_game(game), self.iterations)
        if len(root.children) != 0:
            best_phase = root.best_next_phase
            if best_phase.active_player != player.id:
                return best_phase.active_player
        offset = random.randint(1, game.num_players - 1)
        return (game.active_player + offset) % game.num_players

    def getRecommendedMoves(self, phase, combos):
        root = AlphaZeroNode(
            phase,
            history=[],
            net=self.net,
            prior=1.0,
            trim=self.trim,
            weight=self.weight,
        )
        simulate_node(root, self.iterations)
        scored = []
        for combo in root.next_combos:
            node = root.childmap.get(combo.bitwise)
            # return Combo objects (the unified recommender contract); the caller
            # serializes them via RegiEncoder -> combo_to_dict for the wire
            scored.append((node.visits if node is not None else 0, combo))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [combo for _, combo in scored]
