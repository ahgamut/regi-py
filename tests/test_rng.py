"""The engine RNG is a single reused, seedable thread-local generator."""
import pytest

core = pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

from regi_py.core import GameState, RandomStrategy, GameStatus
from regi_py.logging import DummyLog


def _play(n_players=3):
    game = GameState(DummyLog())
    for _ in range(n_players):
        game.add_player(RandomStrategy())
    game.initialize()
    game.start_loop()
    # a deterministic fingerprint of the whole run
    return (
        game.phase_count,
        len(game.history),
        tuple(p.alive for p in game.players),
        game.history[-1].to_string(),
    )


def _init_random_string(n_players=3):
    game = GameState(DummyLog())
    for _ in range(n_players):
        game.add_player(RandomStrategy())
    game._init_random()
    return game.export_string()


def test_seed_binding_exists():
    assert hasattr(core, "seed")
    core.seed(0)  # callable, no return contract


def test_same_seed_reproduces_full_game():
    core.seed(12345)
    first = _play()
    core.seed(12345)
    again = _play()
    assert first == again


def test_different_seed_diverges():
    core.seed(1)
    a = _play()
    core.seed(2)
    b = _play()
    # astronomically unlikely to coincide across a full game
    assert a != b


def test_seed_reproduces_init_random():
    core.seed(777)
    a = _init_random_string()
    core.seed(777)
    b = _init_random_string()
    assert a == b


def test_seeded_fixture_is_deterministic(seeded):
    # the conftest `seeded` fixture seeds via core.seed; two seedings of the
    # same value reproduce the same random mid-game state
    a = _init_random_string(2)
    core.seed(seeded)
    b = _init_random_string(2)
    assert a == b
