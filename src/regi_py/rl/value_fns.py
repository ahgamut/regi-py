"""Pluggable value-target functions for NN training (torch-free).

The value tensor a net trains on is selected by ``--value-fn``, mirroring the net
registry. A function maps a finished game to a per-training-record value in ``[-1, 1]``
(the Tanh head's range) and is shared by every runner (self-play, brute, team). ``"hp"``
is the default and reproduces the historical discounted-outcome targets exactly.
"""
import math
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from regi_py.rl.utils import hp_loss_penalty, VALUE_DISCOUNT


@dataclass
class ValueContext:
    """What a value function may read about a finished game.

    ``snapshot[positions[k]]`` is training record k's decision phase (by-value
    ``PhaseInfo`` copies); ``len(snapshot) - 1`` is the terminal position.
    ``s0``/``s1`` are total enemy HP at start/end. ``actions`` is parallel to
    ``positions``: ``actions[k]`` is the card LOCATIONS of the combo played at that
    decision (empty for a yield). Actions are known only for RECORDED decisions -- all
    of them in self-play/brute, but only the NN seats' in team games -- so components
    that read actions iterate ``zip(positions, actions)``, not the full ``snapshot``.
    """

    snapshot: List
    positions: List[int]
    win: bool
    s0: float
    s1: float
    actions: List[List[int]] = field(default_factory=list)


# registry: register / get_value_fn / value_fn_names (mirrors the net registry)
_REGISTRY = {}


def register(name):
    def deco(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_value_fn(name):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise SystemExit(f"unknown --value-fn {name!r}; choices: {value_fn_names()}")


def value_fn_names():
    return sorted(_REGISTRY)


def _hp(ctx: ValueContext) -> np.ndarray:
    # reward at the terminal position, discounted by distance-to-end for earlier records
    reward = 1.0 if ctx.win else hp_loss_penalty(ctx.s1)
    last = len(ctx.snapshot) - 1
    return np.array(
        [reward * VALUE_DISCOUNT ** (last - i) for i in ctx.positions],
        dtype=np.float32,
    )


@register("hp")
def hp(ctx: ValueContext) -> np.ndarray:
    """Default and regression anchor: the pre-registry discounted-outcome target."""
    return _hp(ctx)


# --------------------------------------------------------------------------- #
# component functions: each maps a finished game to a per-record RUNNING-PREFIX
# array (one value per ctx.positions entry) in a fixed range. The prefix at record
# k reflects only what happened up to that decision's snapshot position, so an
# intermediate state is credited with the bonuses accrued SO FAR, not the whole
# game's total. Value functions are fixed convex combinations of these (see
# `combine`); a convex combo of [-1, 1] terms stays in [-1, 1] for the Tanh value
# head. Suit index of a card location is ``location // 14`` (CLUBS=0, DIAMONDS=1,
# HEARTS=2, SPADES=3). `NUM_ENEMIES` (12) and the scales below are the fixed spec.
# --------------------------------------------------------------------------- #
NUM_ENEMIES = 12
_SUITS = {"clubs": 0, "diamonds": 1, "hearts": 2, "spades": 3}
_EMPTY_DRAW_SCALE = 20.0
_PACE_RATE = 4.0     # on-pace baseline: expected enemy HP cleared per phase
_PACE_SCALE = 40.0   # tanh scale for the HP-cleared-vs-pace deviation


def _combo_from_locations(locs):
    """Rebuild the played ``Combo`` from its card locations (attacks live in the
    ComboTable, so this resolves; returns ``None`` for a non-cell set e.g. a defense
    combo). Mirrors the ``logs2df`` bitmask->cell->combo pattern."""
    from regi_py.core import ComboTable
    from regi_py.combomap import cell_of_bitwise

    bw = 0
    for loc in locs:
        bw |= 1 << loc
    cell = cell_of_bitwise(bw)
    return ComboTable.make_combo(*cell) if cell is not None else None


def _running(events, positions, denom, lo, hi):
    """Running prefix of per-phase ``events`` (len == len(snapshot)) read at each
    record's snapshot position: ``clip(cumsum(events)[positions] / denom, lo, hi)``.
    ``events[j]`` is the count contributed by phase ``j`` (0 where none). A record at
    position 0 has zero elapsed phases, so its bonus is 0 (also guards the denominator
    against 0/0 once it becomes phase-count-dependent)."""
    idx = np.asarray(positions, dtype=np.intp)
    if idx.size == 0:
        return np.zeros(0, dtype=np.float32)
    csum = np.cumsum(np.asarray(events, dtype=np.float32))
    vals = np.clip(csum[idx] / denom, lo, hi)
    return np.where(idx == 0, np.float32(0.0), vals).astype(np.float32)


def attack_suit_frac(ctx: ValueContext, suit: int) -> np.ndarray:
    """Running min(1, count/12) of recorded ATTACK decisions whose played combo held
    ``suit``, credited at each record."""
    events = np.zeros(len(ctx.snapshot), dtype=np.float32)
    for pos, act in zip(ctx.positions, ctx.actions):
        if ctx.snapshot[pos].phase_attacking and any(loc // 14 == suit for loc in act):
            events[pos] += 1.0
    return _running(events, ctx.positions, NUM_ENEMIES, 0.0, 1.0)


def keep_suit_frac(ctx: ValueContext, suit: int) -> np.ndarray:
    """Running min(1, count/12) of recorded DEFENSE decisions where the defender kept
    >=1 card of ``suit`` (retained it rather than discarding all of it)."""
    events = np.zeros(len(ctx.snapshot), dtype=np.float32)
    for pos, act in zip(ctx.positions, ctx.actions):
        ph = ctx.snapshot[pos]
        if ph.phase_attacking:
            continue
        discarded = set(act)
        kept = (
            card.location
            for card in ph.player_cards[ph.active_player]
            if card.location not in discarded
        )
        if any(loc // 14 == suit for loc in kept):
            events[pos] += 1.0
    return _running(events, ctx.positions, NUM_ENEMIES, 0.0, 1.0)


def exact_kill_frac(ctx: ValueContext) -> np.ndarray:
    """Running min(1, count/12) of attacks that dealt damage exactly equal to the
    enemy's remaining HP."""
    events = np.zeros(len(ctx.snapshot), dtype=np.float32)
    for pos, act in zip(ctx.positions, ctx.actions):
        ph = ctx.snapshot[pos]
        if not ph.phase_attacking or not act or len(ph.enemy_pile) == 0:
            continue
        combo = _combo_from_locations(act)
        if combo is not None and ph.combo_damage(combo) == ph.enemy_pile[0].hp:
            events[pos] += 1.0
    return _running(events, ctx.positions, NUM_ENEMIES, 0.0, 1.0)


def full_block_frac(ctx: ValueContext) -> np.ndarray:
    """Running min(1, count/12) of phases whose accumulated block covers the current
    enemy's attack (enemy hits for 0)."""
    events = np.array(
        [
            1.0
            if len(ph.enemy_pile) and ph.current_block() >= ph.enemy_pile[0].strength
            else 0.0
            for ph in ctx.snapshot
        ],
        dtype=np.float32,
    )
    return _running(events, ctx.positions, NUM_ENEMIES, 0.0, 1.0)


def empty_draw_penalty(ctx: ValueContext) -> np.ndarray:
    """Running max(-1, -(count/20)) over phases with an empty draw pile."""
    events = np.array(
        [1.0 if len(ph.draw_pile) == 0 else 0.0 for ph in ctx.snapshot],
        dtype=np.float32,
    )
    return -_running(events, ctx.positions, _EMPTY_DRAW_SCALE, 0.0, 1.0)


def pacing(ctx: ValueContext) -> np.ndarray:
    """Running pace shaping in (-1, 1): ``tanh((cleared - RATE*p) / SCALE)`` at each
    record's snapshot position ``p``, where ``cleared = s0 - enemy_hp_left`` is total
    enemy HP removed so far. Ahead of the RATE-HP/phase pace -> positive, behind ->
    negative (no win gating). ``p == 0`` (zero elapsed phases) -> 0."""
    out = np.zeros(len(ctx.positions), dtype=np.float32)
    for k, p in enumerate(ctx.positions):
        if p == 0:
            continue
        cleared = ctx.s0 - sum(max(e.hp, 0) for e in ctx.snapshot[p].enemy_pile)
        out[k] = math.tanh((cleared - _PACE_RATE * p) / _PACE_SCALE)
    return out


def combine(ctx: ValueContext, terms) -> np.ndarray:
    """Element-wise convex combination of ``terms`` -> per-record values in [-1, 1].

    ``terms`` is a list of ``(weight, fn)`` where ``fn(ctx)`` returns a scalar (broadcast
    to every record) or a per-record ``np.ndarray`` (e.g. :func:`hp`). Weights should be
    nonnegative and sum to 1 so the result stays in the Tanh head's [-1, 1] range; that
    invariant is the value function author's responsibility (there is no runtime assert)."""
    out = np.zeros(len(ctx.positions), dtype=np.float32)
    for weight, fn in terms:
        out += np.float32(weight) * np.asarray(fn(ctx), dtype=np.float32)
    return out


# --------------------------------------------------------------------------- #
# registered value functions: fixed convex combos (weights sum to 1, so each stays
# in [-1, 1]). All module-level defs so they pickle by qualname across `spawn`.
# --------------------------------------------------------------------------- #
@register("paced")
def paced(ctx: ValueContext) -> np.ndarray:
    return combine(ctx, [(0.8, hp), (0.2, pacing)])


@register("atk")
def atk(ctx: ValueContext) -> np.ndarray:
    return combine(ctx, [(0.8, hp), (0.2, exact_kill_frac)])


@register("atk-blk")
def atk_blk(ctx: ValueContext) -> np.ndarray:
    return combine(ctx, [(0.8, hp), (0.1, exact_kill_frac), (0.1, full_block_frac)])


@register("paced-atk")
def paced_atk(ctx: ValueContext) -> np.ndarray:
    return combine(ctx, [(0.8, hp), (0.15, exact_kill_frac), (0.05, pacing)])


@register("atk-draw")
def atk_draw(ctx: ValueContext) -> np.ndarray:
    return combine(ctx, [(0.8, hp), (0.15, exact_kill_frac), (0.05, empty_draw_penalty)])


@register("paced-blk")
def paced_blk(ctx: ValueContext) -> np.ndarray:
    return combine(ctx, [(0.8, hp), (0.1, pacing), (0.1, full_block_frac)])


def _atk_suit(ctx: ValueContext, suit: int) -> np.ndarray:
    """0.8 hp + 0.10 exact-kill + 0.05 attack-suit + 0.05 keep-suit, for one suit."""
    return combine(
        ctx,
        [
            (0.8, hp),
            (0.10, exact_kill_frac),
            (0.05, lambda c: attack_suit_frac(c, suit)),
            (0.05, lambda c: keep_suit_frac(c, suit)),
        ],
    )


@register("atk-C")
def atk_clubs(ctx: ValueContext) -> np.ndarray:
    return _atk_suit(ctx, _SUITS["clubs"])


@register("atk-D")
def atk_diamonds(ctx: ValueContext) -> np.ndarray:
    return _atk_suit(ctx, _SUITS["diamonds"])


@register("atk-H")
def atk_hearts(ctx: ValueContext) -> np.ndarray:
    return _atk_suit(ctx, _SUITS["hearts"])


@register("atk-S")
def atk_spades(ctx: ValueContext) -> np.ndarray:
    return _atk_suit(ctx, _SUITS["spades"])


def phase_snapshot(phases):
    """By-value ``PhaseInfo`` copies (independent of the engine's history vector)."""
    from regi_py.core import PhaseInfo

    return [PhaseInfo.from_string(p.to_string()) for p in phases]


def assign_values(infos, snapshot, positions, actions, win, s0, s1, value_fn):
    """Set ``info.value`` for each record from ``value_fn`` (infos/positions/actions aligned)."""
    ctx = ValueContext(
        snapshot=snapshot, positions=positions, actions=actions, win=win, s0=s0, s1=s1
    )
    for info, v in zip(infos, value_fn(ctx)):
        info.value = float(v)
    return infos
