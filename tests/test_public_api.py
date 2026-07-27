"""Clean public API names + the thin ``Game`` wrapper.

These lock the additive surface: the clean method names delegate to the same
C++ code as the legacy underscore names, and the ``Game`` wrapper's stepping
loop is byte-for-byte identical to ``start_loop``.
"""
import pytest

import regi_py
import regi_py.core as core
from regi_py.core import GameState, PhaseInfo, RandomStrategy
from regi_py.logging import DummyLog

from conftest import make_game


# --------------------------------------------------------------------------- #
# clean names exist alongside the legacy underscore names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "clean,legacy",
    [
        ("step", "_step"),
        ("set_status", "_set_status"),
        ("init_random", "_init_random"),
        ("init_phaseinfo", "_init_phaseinfo"),
        ("init_string", "_init_string"),
    ],
)
def test_gamestate_clean_and_legacy_names(clean, legacy):
    assert hasattr(GameState, clean)
    assert hasattr(GameState, legacy)  # back-compat: not removed


def test_phaseinfo_clean_and_legacy_names():
    for name in ("randomize", "_randomize", "randomize_from"):
        assert hasattr(PhaseInfo, name)


# --------------------------------------------------------------------------- #
# package exports (regi_py.__all__)
# --------------------------------------------------------------------------- #
def test_package_exports_new_names():
    for name in ("Combo", "LocationInfo", "ComboTable", "seed", "Game"):
        assert hasattr(regi_py, name), name
        assert name in regi_py.__all__


# --------------------------------------------------------------------------- #
# clean names are equivalent to the legacy ones
# --------------------------------------------------------------------------- #
def _play_full(use_clean):
    core.seed(2024)
    g = GameState(DummyLog())
    for _ in range(2):
        g.add_player(RandomStrategy())
    g.initialize()
    n = 0
    while g.is_runnable:
        (g.step if use_clean else g._step)()
        n += 1
    return g.export_string(), n


def test_step_alias_equivalent_to_legacy():
    clean, nc = _play_full(True)
    legacy, nl = _play_full(False)
    assert clean == legacy
    assert nc == nl > 0


def test_init_string_alias_roundtrips(seeded):
    phase = make_game(2).export_phaseinfo()
    s = phase.to_string()
    g1 = GameState(DummyLog())
    g2 = GameState(DummyLog())
    for _ in range(phase.num_players):
        g1.add_player(RandomStrategy())
        g2.add_player(RandomStrategy())
    st1 = g1.init_string(s)
    st2 = g2._init_string(s)
    assert st1 == st2
    assert g1.export_string() == g2.export_string() == s


def test_phaseinfo_randomize_alias_matches_legacy():
    phase = make_game(3).export_phaseinfo()
    a = PhaseInfo.from_string(phase.to_string())
    b = PhaseInfo.from_string(phase.to_string())
    core.seed(7)
    a.randomize(0)
    core.seed(7)
    b._randomize(0)
    assert a.to_string() == b.to_string()


# --------------------------------------------------------------------------- #
# the Game wrapper
# --------------------------------------------------------------------------- #
def test_game_run_matches_start_loop():
    core.seed(99)
    g = regi_py.Game()
    for _ in range(2):
        g.add_player(RandomStrategy())
    g.start()
    g.run()
    via_wrapper = g.export_string()

    core.seed(99)
    raw = GameState(DummyLog())
    for _ in range(2):
        raw.add_player(RandomStrategy())
    raw.initialize()
    raw.start_loop()
    assert via_wrapper == raw.export_string()


def test_game_steps_matches_run():
    core.seed(123)
    g1 = regi_py.Game()
    for _ in range(2):
        g1.add_player(RandomStrategy())
    g1.start()
    steps = 0
    for st in g1.steps():
        assert st is g1.state
        steps += 1
    assert steps > 0

    core.seed(123)
    g2 = regi_py.Game()
    for _ in range(2):
        g2.add_player(RandomStrategy())
    g2.start()
    g2.run()
    assert g1.export_string() == g2.export_string()


def test_game_start_from_string():
    phase = make_game(2).export_phaseinfo()
    s = phase.to_string()
    g = regi_py.Game()
    for _ in range(phase.num_players):
        g.add_player(RandomStrategy())
    g.start(s)
    assert g.export_string() == s


def test_game_delegates_attributes():
    g = regi_py.Game()
    for _ in range(2):
        g.add_player(RandomStrategy())
    g.start()
    # attributes not defined on Game fall through to the wrapped GameState
    assert g.num_players == 2
    assert g.status == g.state.status
    assert isinstance(repr(g), str)
