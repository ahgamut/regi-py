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

import math  # noqa: E402

import regi_py.core as core  # noqa: E402
from regi_py.core import GameState, RandomStrategy, PhaseInfo, Card  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402
from regi_py.strats.phase_utils import PhaseExpander  # noqa: E402

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


def _drive_game(seeded, n_players=3, cap=600):
    """Drive a full game one decision at a time via PhaseExpander (torch-free), always
    playing the first non-yield combo offered. Returns a ValueContext-shaped
    (snapshot, positions, actions, win) -- actions = played card locations per phase."""
    core.seed(seeded)
    game = GameState(DummyLog())
    for _ in range(n_players):
        game.add_player(RandomStrategy())
    game.initialize()
    phase = game.export_phaseinfo()
    snapshot, actions = [], []
    while phase.game_endvalue == 0 and len(snapshot) < cap:
        exp = PhaseExpander(phase)
        offered = exp.offered()
        chosen = next((c for c in offered if c.bitwise != 0), offered[0])
        snapshot.append(PhaseInfo.from_string(phase.to_string()))
        actions.append([c.location for c in chosen.parts])
        phase = PhaseInfo.from_string(exp.step(chosen.bitwise).to_string())
    return snapshot, list(range(len(snapshot))), actions, phase.game_endvalue == 1


def _ctx(seeded):
    snap, pos, acts, win = _drive_game(seeded)
    return vf.ValueContext(snap, pos, actions=acts, win=win, s0=480.0, s1=0.0)


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
    actions = [[] for _ in positions]  # hp ignores actions; any aligned list is fine
    infos = [types.SimpleNamespace(value=None) for _ in positions]
    vf.assign_values(
        infos, snapshot, positions, actions, win=True, s0=480.0, s1=0.0, value_fn=vf.hp
    )
    exp = _expected_hp(len(snapshot), positions, win=True, s1=0.0)
    for info, e in zip(infos, exp):
        assert info.value == float(e)  # stored as a plain float


# --------------------------------------------------------------------------- #
# component functions
# --------------------------------------------------------------------------- #
def test_drive_game_produces_actions(seeded):
    snap, pos, acts, win = _drive_game(seeded)
    assert len(snap) == len(pos) == len(acts) and len(snap) > 0
    # every action card location maps to a suit via loc//14, matching the engine's Card
    for act in acts:
        for loc in act:
            assert loc // 14 == int(Card.from_location(loc).suit)


@pytest.mark.parametrize("suit", [0, 1, 2, 3])
def test_suit_component_ranges(seeded, suit):
    ctx = _ctx(seeded)
    assert 0.0 <= vf.attack_suit_frac(ctx, suit) <= 1.0
    assert 0.0 <= vf.keep_suit_frac(ctx, suit) <= 1.0


def test_attack_suit_frac_matches_independent_count(seeded):
    ctx = _ctx(seeded)
    atk = [ctx.snapshot[p] for p in ctx.positions if ctx.snapshot[p].phase_attacking]
    acts = [a for p, a in zip(ctx.positions, ctx.actions) if ctx.snapshot[p].phase_attacking]
    denom = len(atk)
    for suit in range(4):
        want = (
            sum(any(loc // 14 == suit for loc in a) for a in acts) / denom
            if denom
            else 0.0
        )
        assert vf.attack_suit_frac(ctx, suit) == pytest.approx(want)


def test_scalar_component_ranges(seeded):
    ctx = _ctx(seeded)
    assert 0.0 <= vf.exact_kill_frac(ctx) <= 1.0
    assert 0.0 <= vf.full_block_frac(ctx) <= 1.0
    assert -1.0 <= vf.empty_draw_penalty(ctx) <= 0.0
    assert -1.0 <= vf.pacing(ctx) <= 1.0


def test_empty_context_components_are_zero():
    # no decisions recorded -> action/state ratios collapse to 0 (no div-by-zero)
    ctx = vf.ValueContext([], [], actions=[], win=False, s0=480.0, s1=480.0)
    assert vf.attack_suit_frac(ctx, 0) == 0.0
    assert vf.keep_suit_frac(ctx, 0) == 0.0
    assert vf.exact_kill_frac(ctx) == 0.0
    assert vf.full_block_frac(ctx) == 0.0
    assert vf.empty_draw_penalty(ctx) == 0.0


@pytest.mark.parametrize(
    "n,win,expect",
    [
        (70, True, math.tanh(10 / 15)),   # fast win -> reward
        (70, False, 0.0),                 # fast loss -> neutral
        (79, True, math.tanh(1 / 15)),    # just under the fast threshold
        (80, True, 0.0),                  # boundary: not < 80
        (90, True, 0.0),                  # mid band -> neutral
        (100, False, 0.0),                # boundary: not > 100
        (120, True, -math.tanh(20 / 15)), # long game penalized even on a win
        (120, False, -math.tanh(20 / 15)),
    ],
)
def test_pacing_formula(n, win, expect):
    # pacing reads only len(snapshot) + win, so a length-n dummy snapshot suffices
    ctx = vf.ValueContext([0] * n, list(range(n)), win=win, s0=480.0, s1=0.0)
    assert vf.pacing(ctx) == pytest.approx(expect)


# --------------------------------------------------------------------------- #
# combine (convex combination -> value function)
# --------------------------------------------------------------------------- #
def test_combine_single_term_broadcasts(seeded):
    ctx = _ctx(seeded)
    out = vf.combine(ctx, [(1.0, vf.pacing)])
    assert out.shape == (len(ctx.positions),)
    assert np.allclose(out, np.float32(vf.pacing(ctx)))


def test_combine_mixes_scalar_and_per_position(seeded):
    ctx = _ctx(seeded)
    out = vf.combine(ctx, [(0.5, vf.hp), (0.5, vf.empty_draw_penalty)])
    assert out.shape == (len(ctx.positions),)
    assert out.dtype == np.float32
    assert (out >= -1.0).all() and (out <= 1.0).all()


def test_combine_convex_stays_in_range(seeded):
    ctx = _ctx(seeded)
    comps = [
        vf.exact_kill_frac,
        vf.full_block_frac,
        vf.empty_draw_penalty,
        vf.pacing,
        lambda c: vf.attack_suit_frac(c, 0),
        lambda c: vf.keep_suit_frac(c, 3),
        vf.hp,
    ]
    w = 1.0 / len(comps)
    out = vf.combine(ctx, [(w, f) for f in comps])
    assert (out >= -1.0).all() and (out <= 1.0).all()


# --------------------------------------------------------------------------- #
# registered convex-combo value functions
# --------------------------------------------------------------------------- #
_NEW_VALUE_FNS = [
    "paced", "atk", "atk-blk", "paced-atk", "atk-draw", "paced-blk",
    "atk-C", "atk-D", "atk-H", "atk-S",
]


def test_all_value_fns_registered():
    names = vf.value_fn_names()
    for n in ["hp"] + _NEW_VALUE_FNS:
        assert n in names


@pytest.mark.parametrize("name", _NEW_VALUE_FNS)
def test_value_fn_with_real_actions(seeded, name):
    # exercise the action-reading terms with a real (snapshot, actions) driven game
    ctx = _ctx(seeded)
    out = vf.get_value_fn(name)(ctx)
    assert out.shape == (len(ctx.positions),)
    assert out.dtype == np.float32
    assert (out >= -1.0).all() and (out <= 1.0).all()


def test_value_fn_matches_manual_combo(seeded):
    ctx = _ctx(seeded)
    got = vf.get_value_fn("paced-atk")(ctx)
    want = vf.combine(
        ctx, [(0.8, vf.hp), (0.15, vf.exact_kill_frac), (0.05, vf.pacing)]
    )
    assert np.array_equal(got, want)
