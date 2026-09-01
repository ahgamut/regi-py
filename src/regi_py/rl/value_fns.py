"""Pluggable value-target functions for NN training (torch-free).

The value tensor a net trains on is selected by ``--value-fn``, mirroring the net
registry. A function maps a finished game to a per-training-record value in ``[-1, 1]``
(the Tanh head's range) and is shared by every runner (self-play, brute, team). ``"hp"``
is the default and reproduces the historical discounted-outcome targets exactly.
"""
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from regi_py.rl.utils import hp_loss_penalty, VALUE_DISCOUNT


@dataclass
class ValueContext:
    """What a value function may read about a finished game.

    ``snapshot[positions[k]]`` is training record k's decision phase (by-value
    ``PhaseInfo`` copies); ``len(snapshot) - 1`` is the terminal position.
    ``s0``/``s1`` are total enemy HP at start/end.
    """

    snapshot: List
    positions: List[int]
    win: bool
    s0: float
    s1: float


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


def phase_snapshot(phases):
    """By-value ``PhaseInfo`` copies (independent of the engine's history vector)."""
    from regi_py.core import PhaseInfo

    return [PhaseInfo.from_string(p.to_string()) for p in phases]


def assign_values(infos, snapshot, positions, win, s0, s1, value_fn):
    """Set ``info.value`` for each record from ``value_fn`` (infos and positions aligned)."""
    ctx = ValueContext(snapshot=snapshot, positions=positions, win=win, s0=s0, s1=s1)
    for info, v in zip(infos, value_fn(ctx)):
        info.value = float(v)
    return infos
