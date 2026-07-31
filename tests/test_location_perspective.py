"""Imperfect-information invariant: ``LocationInfo.from_current(phase, cid)`` is
bit-identical before/after ``PhaseInfo.randomize(cid)`` (which reshuffles only what
``cid`` cannot see). Proves a net featurizing via ``from_current`` cannot depend on
the true hidden state.
"""
import random

import numpy as np
import pytest

core = pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

from regi_py.core import GameState, RandomStrategy, BaseStrategy  # noqa: E402
from regi_py.core import LocationInfo, PhaseInfo  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402


def _location_of(phase, cid):
    return np.array(LocationInfo.from_current(phase, cid), dtype=np.uint32)


def _check_phase(phase, stats):
    """For every perspective, the from-current table is unchanged by randomizing
    the hidden cards from that same perspective. Also tallies coverage so the
    assertion can't pass vacuously (empty tables / no hidden cards / no-op
    randomize)."""
    for cid in range(phase.num_players):
        base = _location_of(phase, cid)
        rand = PhaseInfo.randomize_from(phase, cid)
        after = _location_of(rand, cid)
        assert np.array_equal(base, after), (
            f"from_current changed under randomize (perspective {cid})"
        )
        stats["checks"] += 1
        # non-triviality: the randomize actually moved hidden cards ...
        stats["phase_changed"] += int(phase.to_string() != rand.to_string())
        # ... and the view genuinely spreads unknowns across multiple cells
        # (a row with >1 nonzero entry is a card this player can't pin down)
        stats["unknown_rows"] += int(np.sum(np.count_nonzero(base, axis=1) > 1) > 0)


class _InvariantStrategy(BaseStrategy):
    """Plays randomly, but checks the perspective invariant on the live phase at
    every attack/defense decision (diverse, deep-in-game states). Subclasses
    ``BaseStrategy`` (not the C++ ``RandomStrategy``, whose methods don't route
    back to Python overrides)."""

    __strat_name__ = "location-perspective-invariant"

    def __init__(self, stats):
        super().__init__()
        self.stats = stats

    def setup(self, player, game):
        return 0

    def getRedirectIndex(self, player, game):
        return 0

    def getAttackIndex(self, combos, player, yield_allowed, game):
        _check_phase(game.export_phaseinfo(), self.stats)
        return random.randint(0, len(combos) - 1) if combos else -1

    def getDefenseIndex(self, combos, player, damage, game):
        _check_phase(game.export_phaseinfo(), self.stats)
        return random.randint(0, len(combos) - 1) if combos else -1


def test_from_current_invariant_under_randomize_midgame(seeded):
    """Fresh random mid-game states (``init_random``), every perspective."""
    random.seed(seeded)
    stats = dict(checks=0, phase_changed=0, unknown_rows=0)
    for _ in range(120):
        game = GameState(DummyLog())
        for _ in range(random.choice([2, 3, 4])):
            game.add_player(RandomStrategy())
        game.init_random()
        _check_phase(game.export_phaseinfo(), stats)

    assert stats["checks"] > 200, stats
    # every mid-game state has hidden cards to move and unknowns to spread
    assert stats["phase_changed"] == stats["checks"], stats
    assert stats["unknown_rows"] == stats["checks"], stats


def test_from_current_invariant_under_randomize_fullgames(seeded):
    """The invariant at every real decision node across many full games."""
    random.seed(seeded + 1)
    stats = dict(checks=0, phase_changed=0, unknown_rows=0)
    for _ in range(40):
        game = GameState(DummyLog())
        for _ in range(random.choice([2, 3, 4])):
            game.add_player(_InvariantStrategy(stats))
        game.initialize()
        game.start_loop()

    # the equality assertion fires inside the strategy; ensure it was exercised
    # across many non-trivial states (real hidden cards, real spread unknowns)
    assert stats["checks"] > 500, stats
    assert stats["phase_changed"] > 0, stats
    assert stats["unknown_rows"] > 0, stats
