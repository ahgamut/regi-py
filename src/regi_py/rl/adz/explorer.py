"""AlphaDouZero (ADZ) MCTS node + playable strategies.

The AZ analogue lives in ``az/explorer.py``; this is the forked candidate-scoring
stack. ``ADZNode`` subclasses ``strats.mcts_explorer.MCTSNode`` (reusing the lazy
``PhaseExpander`` expansion and the base value accounting) but reads its child
priors from a ``CandidateBaseNet`` (``net.predict(history, offered, phase)`` ->
per-offered-subset priors) instead of the AZ ``(56,22)`` grid, so attack AND
defense decisions are both scored. The node is AGNOSTIC to the net's action
encoding: it stores the offered ``bitwise``s, a net-agnostic per-candidate feature
block (captured live via ``candidate_semantics`` -- defense combos can't be rebuilt
from a bare ``bitwise``), and the visit policy; the net decides membership.

``ADZNodeInfo`` holds plain values only (the dangling-``PhaseInfo`` rule: ``history``
is by-value ``export_phaseinfo`` copies threaded through the tree), and ``export``
never touches a grid or ``cell_of_bitwise``.
"""
from regi_py.core import PhaseInfo, BaseStrategy
from regi_py.strats.mcts_explorer import MCTSNode
from regi_py.strats.recommender import RecommenderMixin
from regi_py.strats.phase_utils import index_of_bitwise
from regi_py.rl.features import candidate_semantics, CAND_FEATURE_DIM
from regi_py.rl.utils import *

#
import random
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any


def trimmed_history(history, phase, maxhist):
    """The last ``maxhist`` phases ending at ``phase``, left-padded with the oldest
    frame when short (same window contract as ``AlphaZeroNode._trimmed_history``)."""
    tmp = list(history) + [phase]
    if len(tmp) >= maxhist:
        return tmp[-maxhist:]
    n = len(tmp)
    return [tmp[0]] * (maxhist - n) + tmp


@dataclass(slots=True)
class ADZNodeInfo:
    history: Tuple[PhaseInfo]     # by-value phase copies (dangling-reference rule)
    candidates: Any              # list of offered combos' bitwise (u64)
    cand_feats: Any              # (K, F) candidate_semantics, captured live
    policy: Any                  # (K,) visit fractions over the offered subsets
    value: float                 # z placeholder; the self-play driver overwrites it
    attacking: float             # 0/1 phase flag (kept for analysis; loss is unmasked)


class ADZNode(MCTSNode):
    rng = np.random.default_rng()

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
        self.history = trimmed_history(history, self.root_phase, self.net.max_history)
        # captured live so export() has it; None for a terminal leaf
        self.cand_feats = None
        #
        # a terminal leaf has no children; simulate() returns the endgame reward,
        # so the net eval is never read -- skip it
        if self.root_phase.game_endvalue != 0:
            self.leaf_value = 0.0
            self.next_priors = np.zeros(0, dtype=np.float32)
            return
        #
        v_hat, priors = self.net.predict(self.history, self.next_combos, self.root_phase)
        # net value estimate for this leaf; kept separate from ``self.value`` (the
        # base MCTSNode's backed-up reward sum W), exactly as AlphaZeroNode does
        self.leaf_value = v_hat
        self.cand_feats = candidate_semantics(self.root_phase, self.next_combos)
        n = len(self.next_combos)
        self.next_priors = np.zeros(n, dtype=np.float32)
        if n != 0:
            self.next_priors[: len(priors)] = priors
            self.next_priors = normalize_probs(1e-3 + self.next_priors)
            # order expansion by prior once, ascending, so expand() pop()s the
            # highest-prior child first (mirrors AlphaZeroNode)
            self.rem_exp_ind.sort(key=lambda j: self.next_priors[j])

    def add_dirichlet_noise(self, alpha=0.35, frac=0.2):
        """Mix Dirichlet exploration noise into this node's child priors. Applied
        only at the search root (self-play drivers call it per move). Unlike the AZ
        node this fires for EVERY decision, attack and defense, because every ADZ
        node carries net priors over its offered subsets."""
        n = len(self.next_combos)
        if n == 0:
            return
        noise = self.rng.dirichlet([alpha] * n)
        self.next_priors = normalize_probs(
            (1 - frac) * self.next_priors + frac * noise
        )

    @property
    def ucb1(self):
        # PUCT: Q + weight * prior * sqrt(N_parent) / (1 + n), same as AlphaZeroNode
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
        # the highest-prior unexpanded child first (net policy orders exploration)
        i = self.rem_exp_ind.pop()
        combo = self.next_combos[i]
        prior = self.next_priors[i]
        phase = self._expander.step(combo.bitwise)
        new_node = ADZNode(
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
        # keep backups on the same [-1, 1] scale as the tanh value head and the z
        # training target: win -> 1.0, loss -> shaped negative, else net estimate
        if end_value == 1:
            return 1.0
        if end_value == -1:
            return hp_loss_penalty(enemy_hp_left(self.root_phase))
        return self.leaf_value

    def export(self):
        N0 = self.visits
        n = len(self.next_combos)
        candidates = [combo.bitwise for combo in self.next_combos]
        policy = np.zeros(n, dtype=np.float32)
        for i, combo in enumerate(self.next_combos):
            node = self.childmap.get(combo.bitwise)
            policy[i] = node.visits if node is not None else 0
        if N0 > 0:
            policy /= N0  # visit fractions in [0, 1] (matches the masked softmax target)
        cand_feats = (
            self.cand_feats
            if self.cand_feats is not None
            else np.zeros((n, CAND_FEATURE_DIM), dtype=np.float32)
        )
        return ADZNodeInfo(
            history=self.history,
            candidates=candidates,
            cand_feats=cand_feats,
            # placeholder: the self-play driver overwrites every info.value with the
            # discounted whole-game outcome z
            policy=policy,
            value=0.0,
            attacking=float(self.root_phase.phase_attacking),
        )


def adz_simulate_node(root_node, iterations):
    """Run ``iterations`` of candidate-scored MCTS from ``root_node`` (in place)."""
    for _ in range(iterations):
        node = ADZNode.select(root_node)
        if not node.is_terminal():
            node = node.expand()
        reward = node.simulate()
        ADZNode.update(node, reward)
    return root_node


def _random_redirect(game):
    offset = random.randint(1, game.num_players - 1)
    return (game.active_player + offset) % game.num_players


class ADZDirectStrategy(BaseStrategy, RecommenderMixin):
    """ADZ net policy with NO search: play the highest-prior offered subset. The
    analogue of ``az.explorer.NetDirectStrategy``. ``getRedirectIndex`` is out of
    scope for the candidate stack, so it just picks a random other player."""

    __strat_name__ = "adz-direct"
    num_recos = 5

    def __init__(self, net):
        super(ADZDirectStrategy, self).__init__()
        self.net = net
        self.__strat_name__ = f"adz-direct-{net.__mname__}"

    def setup(self, player, game):
        self.net.eval()
        return 0

    def getRedirectIndex(self, player, game):
        return _random_redirect(game)

    def getRecommendedMoves(self, phase, combos):
        # rank the offered subsets by the net's priors (no search); returns Combo
        # objects, the unified recommender contract
        if len(combos) == 0:
            return []
        history = trimmed_history([], phase, self.net.max_history)
        _, priors = self.net.predict(history, combos, phase)
        order = np.argsort(priors)[::-1][: self.num_recos]
        return [combos[int(i)] for i in order]

    def _policy_index(self, combos, game):
        if len(combos) == 0:
            return -1
        root_phase = game.export_phaseinfo()
        history = trimmed_history(game.history, root_phase, self.net.max_history)
        _, priors = self.net.predict(history, combos, root_phase)
        return int(np.argmax(priors))

    def getAttackIndex(self, combos, player, yield_allowed, game):
        return self._policy_index(combos, game)

    def getDefenseIndex(self, combos, player, damage, game):
        return self._policy_index(combos, game)


class ADZExplorerStrategy(BaseStrategy, RecommenderMixin):
    """Candidate-scored MCTS as a playable strategy (the ADZ analogue of
    ``AZExplorerStrategy``): run search from the current state and play the
    most-visited offered subset. No Dirichlet noise (competitive play)."""

    __strat_name__ = "adz-explorer"

    def __init__(self, net, iterations=64, trim=True, weight=math.sqrt(2)):
        super(ADZExplorerStrategy, self).__init__()
        self.net = net
        self.iterations = iterations
        self.trim = trim
        self.weight = weight
        self.__strat_name__ = f"adz-{net.__mname__}-{iterations}"

    def setup(self, player, game):
        self.net.eval()
        return 0

    def _root_from_game(self, game):
        return ADZNode(
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
        root = adz_simulate_node(self._root_from_game(game), self.iterations)
        if len(root.children) == 0:
            return -1
        return index_of_bitwise(combos, root.best_combo.bitwise)

    def getAttackIndex(self, combos, player, yield_allowed, game):
        return self._search_index(game, combos)

    def getDefenseIndex(self, combos, player, damage, game):
        return self._search_index(game, combos)

    def getRedirectIndex(self, player, game):
        return _random_redirect(game)

    def getRecommendedMoves(self, phase, combos):
        root = ADZNode(
            phase,
            history=[],
            net=self.net,
            prior=1.0,
            trim=self.trim,
            weight=self.weight,
        )
        adz_simulate_node(root, self.iterations)
        scored = []
        for combo in root.next_combos:
            node = root.childmap.get(combo.bitwise)
            # return Combo objects (the unified recommender contract)
            scored.append((node.visits if node is not None else 0, combo))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [combo for _, combo in scored]
