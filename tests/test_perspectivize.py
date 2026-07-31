"""``regi_py.rl.utils.perspectivize`` leaves the three observables a net consumes
UNCHANGED -- the used-pile ``ComboTable``, the active player's own hand, and the
``LocationInfo.from_current`` table -- while reshuffling the hidden cards.

``perspectivize`` lives in torch-free ``regi_py.rl.utils``, but importing the
``regi_py.rl`` package runs its ``__init__`` (torch nets), so -- exactly as
``test_adz_explorer`` does -- we shadow ``regi_py.rl`` with a torch-free stub when
the real (torch) import is unavailable.
"""
import sys
import types
import pathlib
import random

import numpy as np
import pytest

pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

_RL_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "regi_py" / "rl"
if "regi_py.rl" not in sys.modules:
    try:
        import regi_py.rl  # noqa: F401  (real package; succeeds under torch)
    except Exception:
        _stub = types.ModuleType("regi_py.rl")
        _stub.__path__ = [str(_RL_DIR)]
        sys.modules["regi_py.rl"] = _stub

from regi_py.core import GameState, RandomStrategy  # noqa: E402
from regi_py.core import LocationInfo, ComboTable  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402
from regi_py.rl.utils import perspectivize  # noqa: E402


def _hand(phase):
    """Sorted card locations of the phase's active player's own hand."""
    return sorted(c.location for c in phase.player_cards[phase.active_player])


def _used_pile(phase):
    return np.array(ComboTable.from_phase(phase), dtype=np.uint32)


def _location(phase):
    return np.array(
        LocationInfo.from_current(phase, phase.active_player), dtype=np.uint32
    )


def test_perspectivize_preserves_observables(seeded):
    random.seed(seeded)
    stats = dict(checks=0, phase_changed=0)
    for _ in range(200):
        game = GameState(DummyLog())
        for _ in range(random.choice([2, 3, 4])):
            game.add_player(RandomStrategy())
        game.init_random()
        phase = game.export_phaseinfo()

        before_hand = _hand(phase)
        before_used = _used_pile(phase)
        before_loc = _location(phase)

        view = perspectivize(phase)

        # the active player's own hand is untouched
        assert _hand(view) == before_hand
        # the used pile (public) is untouched
        assert np.array_equal(_used_pile(view), before_used)
        # the perspective location table is untouched
        assert np.array_equal(_location(view), before_loc)
        # perspectivize returns a copy: the original phase is not mutated
        assert _hand(phase) == before_hand
        assert np.array_equal(_used_pile(phase), before_used)

        stats["checks"] += 1
        # non-triviality: the hidden cards really were reshuffled
        stats["phase_changed"] += int(phase.to_string() != view.to_string())

    assert stats["checks"] == 200
    # every mid-game state has hidden cards to move, so the view must differ from
    # the omniscient phase -- otherwise the invariance above would be vacuous
    assert stats["phase_changed"] == stats["checks"], stats
