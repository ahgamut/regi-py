"""Shared pytest fixtures for the regi_py test suite.

The C++ extension ``regi_py.core`` must be built for the running interpreter
(``pip install -e . --no-build-isolation``).  These fixtures build small games so
individual tests do not repeat the boilerplate of adding players and initializing.
"""
import pytest

# Import the extension eagerly so a build/ABI mismatch fails collection with a
# clear message rather than deep inside a test.
core = pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

from regi_py.core import GameState, RandomStrategy  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402


def _seed(value=1234):
    """Seed the engine RNG when a seed() binding exists."""
    seed_fn = getattr(core, "seed", None)
    if seed_fn is not None:
        seed_fn(value)


@pytest.fixture
def seeded():
    """Seed the RNG (no-op until the seed() binding lands) and return the seed."""
    _seed()
    return 1234


def make_game(num_players=2, strategies=None, log=None):
    """Build and randomly initialize a GameState with ``num_players`` players.

    Returns the running GameState.  ``strategies`` may be a list of strategy
    instances; otherwise RandomStrategy is used for every seat.
    """
    log = log if log is not None else DummyLog()
    game = GameState(log)
    if strategies is None:
        strategies = [RandomStrategy() for _ in range(num_players)]
    for strat in strategies:
        game.add_player(strat)
    game._init_random()
    return game


@pytest.fixture
def game():
    """A freshly randomized 2-player game (RandomStrategy)."""
    return make_game(2)


@pytest.fixture
def phase(game):
    """A PhaseInfo exported from a fresh 2-player game."""
    return game.export_phaseinfo()
