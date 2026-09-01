"""ADZ-specific self-play data recorders + training-record builder.

The GAME RUNNERS / eval (self-play, brute, team, ``test_model``,
``improved_gameplay``) are now the paradigm-agnostic ones in ``rl.training``, driven
by a ``trainer_loop.Paradigm`` (the ADZ instance is built in the unified
``trainers/trainer.py``). What stays ADZ-specific lives here: the two brute/team
recorders (candidate-scoring: they must capture each decision's offered list +
per-candidate semantics, not just a played combo) and ``adz_infos_from_game`` (the
one-shot ``ADZNodeInfo`` builder). The recorders reuse the shared
``_BruteRecordMixin`` / ``_TeamRecordMixin`` from ``rl.training``.
"""
import numpy as np

from regi_py.core import MAX_CARDS_IN_GAME
from regi_py.strats import BruteSamplingStrategy
from regi_py.rl.features import candidate_semantics
from regi_py.rl.adz.explorer import ADZNodeInfo, ADZExplorerStrategy, trimmed_history
from regi_py.rl.training import _BruteRecordMixin, _TeamRecordMixin
from regi_py.rl.value_fns import assign_values, phase_snapshot


class RecordingADZBruteStrategy(_BruteRecordMixin, BruteSamplingStrategy):
    """Brute self-play recorder (ADZ). Unlike the AZ recorder it must capture each
    decision's full offered list + per-candidate semantics (the net scores the ragged
    offer, not a fixed grid), computed from the by-value ``export_phaseinfo`` copy and
    kept as plain numpy/ints; window rebuilt post-game in ``adz_infos_from_game``."""

    def __init__(self, iterations=64):
        super().__init__(iterations=iterations)
        # (history_index, [bitwise...], cand_feats(K,F), played_bitwise, attacking)
        self.moves = []

    def _append(self, root_phase, combos, ind, game):
        combo = combos[ind]
        idx = len(game.history) - 1  # decision phase is history[-1] (plain int)
        bitwises = [c.bitwise for c in combos]
        feats = candidate_semantics(root_phase, combos)  # plain np, safe to hold
        self.moves.append(
            (idx, bitwises, feats, combo.bitwise, float(root_phase.phase_attacking))
        )


def adz_infos_from_game(game, moves, win, s0, s1, net_cls, value_fn):
    """Build the ``ADZNodeInfo`` training list AFTER a brute/team game finishes, from
    the now-stable ``game.history``. One-shot analogue of ``ADZNode.export()``: the
    policy is a one-hot at the played subset over that decision's offered list.
    ``value_fn`` sets each value; ``moves``' history indices are the snapshot positions."""
    hist = list(game.history)
    maxhist = net_cls.max_history
    infos = []
    positions = []
    actions = []
    for idx, bitwises, feats, played_bw, attacking in moves:
        root_phase = hist[idx]
        window = trimmed_history(hist[:idx], root_phase, maxhist)
        K = len(bitwises)
        policy = np.zeros(K, dtype=np.float32)
        pi = bitwises.index(played_bw) if played_bw in bitwises else 0
        if K:
            policy[pi] = 1.0
        infos.append(
            ADZNodeInfo(
                history=window,
                candidates=bitwises,
                cand_feats=feats,
                policy=policy,
                value=0.0,
                attacking=attacking,
            )
        )
        positions.append(idx)
        # the played combo's card locations = set bits of its location bitmask
        actions.append([b for b in range(MAX_CARDS_IN_GAME) if (played_bw >> b) & 1])
    assign_values(infos, phase_snapshot(hist), positions, actions, win, s0, s1, value_fn)
    return infos


class RecordingADZTeamStrategy(_TeamRecordMixin, ADZExplorerStrategy):
    """``ADZExplorerStrategy`` that records each of its own decisions the way an
    ``ADZNodeInfo`` needs (``(history_index, [bitwise...], cand_feats, played_bitwise,
    attacking)``), rebuilt post-game by ``adz_infos_from_game``. See
    the team-games design notes."""

    def __init__(self, net, iterations, moves):
        super().__init__(net, iterations=iterations, trim=False)
        self.moves = moves

    def _record(self, combos, ind, game):
        if ind is None or ind < 0 or not combos:
            return
        root_phase = game.export_phaseinfo()  # by-value copy: safe to featurize now
        idx = len(game.history) - 1
        bitwises = [c.bitwise for c in combos]
        feats = candidate_semantics(root_phase, combos)
        self.moves.append(
            (idx, bitwises, feats, combos[ind].bitwise, float(root_phase.phase_attacking))
        )
