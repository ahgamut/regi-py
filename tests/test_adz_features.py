"""Torch-free tests for the ADZ candidate featurizers in ``rl/features.py``.

``features.py`` is deliberately torch-free (``regi_py.core`` + numpy + combomap
only), but importing the ``regi_py.rl`` package runs its ``__init__`` which pulls in
the torch-dependent nets. To keep these runnable wherever the C++ extension is built
(torch NOT required), load ``features.py`` directly by path.

Covers:
- ``candidate_semantics`` -- per-offered-subset feature block; columns agree with the
  phase's own combo-effect methods on real attack AND defense decisions.
- ``keepy_marginal`` -- the derived per-card "kept" CFR hint.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

from regi_py.core import GameState, RandomStrategy, PhaseInfo  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402
from regi_py.strats.phase_utils import PhaseExpander  # noqa: E402

# load rl/features.py in isolation (no torch, no rl/__init__)
_FEATURES = pathlib.Path(__file__).resolve().parents[1] / "src" / "regi_py" / "rl" / "features.py"
_spec = importlib.util.spec_from_file_location("_adz_features_under_test", _FEATURES)
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


def _collect_phases(seeded, n_games=25):
    """Attack and defense decision-phase strings from real games (stable copies)."""
    import regi_py.core as core

    core.seed(seeded)
    attack, defense = [], []
    for _ in range(n_games):
        game = GameState(DummyLog())
        for _ in range(2):
            game.add_player(RandomStrategy())
        game.initialize()
        game.start_loop()
        for ph in game.history:  # read while the game is alive; copy via to_string
            (attack if ph.phase_attacking else defense).append(ph.to_string())
    return attack, defense


def test_candidate_semantics_matches_phase_methods(seeded):
    attack, defense = _collect_phases(seeded)
    assert attack and defense, "expected both attack and defense decisions"

    checked = dict(rows=0, lethal=0, block=0, damage=0)
    for s in attack[:150] + defense[:150]:
        phase = PhaseInfo.from_string(s)
        combos = PhaseExpander(phase).offered()
        if not combos:
            continue
        F = features.candidate_semantics(phase, combos)
        assert F.shape == (len(combos), features.CAND_FEATURE_DIM)
        assert F.dtype == np.float32 and np.isfinite(F).all()
        assert (F[:, 4] >= 0).all() and (F[:, 4] <= 1).all()  # can_attack is 0/1

        enemy_hp = phase.enemy_pile[0].hp if len(phase.enemy_pile) else 0
        for i, c in enumerate(combos):
            dmg, blk = phase.combo_damage(c), phase.combo_block(c)
            assert F[i, 0] == pytest.approx(dmg / features.CAP_SCALE)
            assert F[i, 1] == pytest.approx(blk / features.CAP_SCALE)
            assert F[i, 2] == pytest.approx(c.base_damage / features.CAP_SCALE)
            assert F[i, 3] == pytest.approx(c.base_defense / features.CAP_SCALE)
            assert F[i, 4] == float(bool(c.can_attack))
            assert F[i, 5] == pytest.approx(len(c.parts) / features._PARTS_SCALE)
            assert F[i, 6] == float(c.bitwise == 0)
            assert F[i, 7] == float(enemy_hp > 0 and dmg >= enemy_hp)
            assert 0.0 <= F[i, 8] <= 1.0
            checked["rows"] += 1
            checked["lethal"] += int(F[i, 7] == 1.0)
            checked["block"] += int(blk != 0)
            checked["damage"] += int(dmg != 0)

    # not vacuous: real damage, real blocks and some lethal moves were exercised
    assert checked["rows"] > 500, checked
    assert checked["lethal"] > 0 and checked["block"] > 0 and checked["damage"] > 0, checked


def test_candidate_semantics_empty_list():
    """K == 0 yields a well-shaped empty block, not an error."""
    game = GameState(DummyLog())
    for _ in range(2):
        game.add_player(RandomStrategy())
    game.initialize()
    F = features.candidate_semantics(game.export_phaseinfo(), [])
    assert F.shape == (0, features.CAND_FEATURE_DIM) and F.dtype == np.float32


def test_keepy_marginal_basic():
    # combos: {0,1} @ .5, {1,2} @ .5, yield @ 0  -> played 0:.5 1:1 2:.5
    km = features.keepy_marginal([0b0011, 0b0110, 0], [0.5, 0.5, 0.0])
    assert km.shape == (56,) and km.dtype == np.float32
    assert km[0] == pytest.approx(0.5)
    assert km[1] == pytest.approx(0.0)   # in every played subset -> never kept
    assert km[2] == pytest.approx(0.5)
    assert km[3:].min() == 1.0           # untouched cards are fully kept
    assert km.min() >= 0.0 and km.max() <= 1.0


def test_keepy_marginal_matches_membership_on_real_offer(seeded):
    """On a real offered list with a uniform policy, keepyness equals
    1 − (fraction of subsets each card appears in)."""
    attack, _ = _collect_phases(seeded, n_games=5)
    phase = PhaseInfo.from_string(attack[len(attack) // 2])
    combos = PhaseExpander(phase).offered()
    bitwises = [c.bitwise for c in combos]
    policy = np.full(len(combos), 1.0 / len(combos), dtype=np.float32)

    km = features.keepy_marginal(bitwises, policy)
    # independent recomputation of the played marginal
    played = np.zeros(56, dtype=np.float64)
    for p, bw in zip(policy, bitwises):
        for loc in range(56):
            if (bw >> loc) & 1:
                played[loc] += p
    assert np.allclose(km, np.clip(1.0 - played, 0.0, 1.0), atol=1e-6)
