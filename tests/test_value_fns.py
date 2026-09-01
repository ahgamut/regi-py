"""Torch-free tests for the pluggable value-target registry (``rl/value_fns.py``).

``value_fns`` is torch-free (it needs only ``rl.utils`` + numpy + the engine), but
importing the ``regi_py.rl`` package runs its ``__init__`` which pulls in the torch
nets. Same trick as ``test_adz_explorer``: shadow ``regi_py.rl`` with a torch-free stub
package so its submodules import without the eager net imports.

The load-bearing check is that ``"hp"`` (the default) reproduces the historical
``reward * VALUE_DISCOUNT ** (distance-to-end)`` targets exactly, so wiring the registry
into the runners changed no training target.
"""
import sys
import types
import pathlib

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

import regi_py.core as core  # noqa: E402
from regi_py.core import GameState, RandomStrategy, PhaseInfo  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402

import regi_py.rl.value_fns as vf  # noqa: E402
from regi_py.rl.utils import hp_loss_penalty, VALUE_DISCOUNT  # noqa: E402


def _game_history(seeded, n_players=3):
    """A real game's decision phases (stable by-value copies)."""
    core.seed(seeded)
    game = GameState(DummyLog())
    for _ in range(n_players):
        game.add_player(RandomStrategy())
    game.initialize()
    game.start_loop()
    return vf.phase_snapshot(game.history), game


def _expected_hp(n_snapshot, positions, win, s1):
    """The pre-registry formula, recomputed the same way _hp builds its array."""
    reward = 1.0 if win else hp_loss_penalty(s1)
    last = n_snapshot - 1
    return np.array(
        [reward * VALUE_DISCOUNT ** (last - p) for p in positions], dtype=np.float32
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_lists_hp_default():
    assert "hp" in vf.value_fn_names()
    assert vf.get_value_fn("hp") is vf.hp


def test_get_value_fn_unknown_raises():
    with pytest.raises(SystemExit):
        vf.get_value_fn("does-not-exist")


# --------------------------------------------------------------------------- #
# "hp" is the regression anchor
# --------------------------------------------------------------------------- #
def test_hp_dense_win(seeded):
    snapshot, _ = _game_history(seeded)
    positions = list(range(len(snapshot)))  # self-play: one record per phase
    ctx = vf.ValueContext(snapshot, positions, win=True, s0=480.0, s1=0.0)
    got = vf.hp(ctx)
    exp = _expected_hp(len(snapshot), positions, win=True, s1=0.0)
    assert got.dtype == np.float32
    assert np.array_equal(got, exp)
    assert got[-1] == np.float32(1.0)  # terminal position keeps full reward


@pytest.mark.parametrize("s1", [0.0, 150.0, 290.0, 400.0])  # one per hp_loss_penalty tier
def test_hp_sparse_loss(seeded, s1):
    snapshot, _ = _game_history(seeded)
    n = len(snapshot)
    positions = [i for i in range(n) if i % 3 == 0]  # team-like: sparse-but-true indices
    ctx = vf.ValueContext(snapshot, positions, win=False, s1=s1, s0=480.0)
    got = vf.hp(ctx)
    exp = _expected_hp(n, positions, win=False, s1=s1)
    assert np.array_equal(got, exp)
    assert len(got) == len(positions)


# --------------------------------------------------------------------------- #
# generic contract every registered fn must honor (future-proofs new fns)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", vf.value_fn_names())
@pytest.mark.parametrize("win", [True, False])
def test_contract_shape_and_range(seeded, name, win):
    snapshot, _ = _game_history(seeded)
    positions = list(range(len(snapshot)))
    ctx = vf.ValueContext(snapshot, positions, win=win, s1=120.0, s0=480.0)
    out = vf.get_value_fn(name)(ctx)
    assert out.shape == (len(positions),)
    assert (out >= -1.0).all() and (out <= 1.0).all()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_phase_snapshot_is_faithful_and_independent(seeded):
    snapshot, game = _game_history(seeded)
    hist = list(game.history)
    assert len(snapshot) == len(hist)
    for copy, orig in zip(snapshot, hist):
        assert isinstance(copy, PhaseInfo)
        assert copy.to_string() == orig.to_string()  # exact copy
    # independent of the engine: dropping the game leaves the copies intact
    first = snapshot[0].to_string()
    del game, hist
    assert snapshot[0].to_string() == first


def test_assign_values_sets_info_value(seeded):
    snapshot, _ = _game_history(seeded)
    positions = list(range(len(snapshot)))
    infos = [types.SimpleNamespace(value=None) for _ in positions]
    vf.assign_values(infos, snapshot, positions, win=True, s0=480.0, s1=0.0, value_fn=vf.hp)
    exp = _expected_hp(len(snapshot), positions, win=True, s1=0.0)
    for info, e in zip(infos, exp):
        assert info.value == float(e)  # stored as a plain float
