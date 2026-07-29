"""Shared, architecture-independent featurization for the AlphaZero nets.

Every net (`nets/*.py`) turns the same raw per-phase arrays into its own input
layout, so the expensive, cached featurization lives here ONCE and each net only
defines how it *assembles* these arrays (see ``nets/base.py`` ``_assemble``).

Raw per-phase arrays (on the 56-card-location axis):
  location   (56, 9)   L1 row-normalized ``LocationInfo.from_current`` (perspective-dependent)
  used_pile  (56, 22)  ``ComboTable.from_phase``
  capability (56, 2)   signed per-card [attack, defense], scaled by 1/CAP_SCALE

Targets (also architecture-independent): value, keepyness (56,), atk_probs
(56, 22), attacking -- see ``shared_targets``.
"""
from regi_py.core import LocationInfo, ComboTable, Card
from regi_py.core import MAX_CARDS_IN_GAME, MAX_LOCATIONS, MAX_PLAYED_STATUS

import numpy as np

# per-card capability channels: [attack_capability, defense_capability]
CAP_CHANNELS = 2
# scale factor keeping capabilities roughly in [-1, 1]: a King has 40 HP -> -1.0
# and deals 20 base damage -> -0.5
CAP_SCALE = 40.0

# static raw per-card strength by location; enemy-pile cards override this each
# phase (their HP / base damage), so only the non-enemy part is precomputed
_STRENGTH = np.zeros(MAX_CARDS_IN_GAME, dtype=np.float32)
for _loc in range(MAX_CARDS_IN_GAME):
    try:
        _STRENGTH[_loc] = Card.from_location(_loc).strength
    except Exception:
        pass


def card_capabilities(phase):
    """Per-card ``(attack, defense)`` capability for ``phase``, on the 56-location
    axis. Shape ``(MAX_CARDS_IN_GAME, CAP_CHANNELS)``, scaled by ``1/CAP_SCALE``.

    A card *not* in the enemy pile contributes its own (non-negative) strength to
    both channels. A card *in* the enemy pile is a target, encoded negatively:
    attack = ``-max(0, current HP)``, defense = ``-(base damage it deals)``.
    """
    attack = _STRENGTH.copy()
    defense = _STRENGTH.copy()
    for enemy in phase.enemy_pile:
        loc = enemy.location
        attack[loc] = -max(0, enemy.hp)
        defense[loc] = -enemy.strength
    caps = np.empty((MAX_CARDS_IN_GAME, CAP_CHANNELS), dtype=np.float32)
    caps[:, 0] = attack
    caps[:, 1] = defense
    caps /= CAP_SCALE
    return caps


# content-keyed caches (by phase.to_string()) so the rolling history window and
# sibling MCTS nodes don't re-tensorize the same phase repeatedly. The caller
# computes `pstr = phase.to_string()` ONCE per phase and threads it into all
# three helpers -- the serialization is the dominant per-lookup cost.
_CACHE_CAP = 8192
_LOC_CACHE = {}  # (phase_str, perspective) -> np (56, 9)
_USP_CACHE = {}  # phase_str -> np (56, 22)
_CAP_CACHE = {}  # phase_str -> np (56, CAP_CHANNELS)


def _cache_put(cache, key, val):
    if len(cache) >= _CACHE_CAP:
        cache.clear()
    cache[key] = val
    return val


def _location_array(phase, perspective, pstr):
    key = (pstr, perspective)
    a = _LOC_CACHE.get(key)
    if a is None:
        loca0 = np.array(LocationInfo.from_current(phase, perspective), dtype=np.float32)
        a = _cache_put(_LOC_CACHE, key, loca0 / loca0.sum(axis=1, keepdims=True))
    return a


def _used_pile_array(phase, pstr):
    a = _USP_CACHE.get(pstr)
    if a is None:
        a = _cache_put(_USP_CACHE, pstr, np.array(ComboTable.from_phase(phase), dtype=np.float32))
    return a


def _capability_array(phase, pstr):
    a = _CAP_CACHE.get(pstr)
    if a is None:
        a = _cache_put(_CAP_CACHE, pstr, card_capabilities(phase))
    return a


def raw_phase_arrays(phase, perspective):
    """The three raw per-card arrays for one phase: ``(loc(56,9), usp(56,22),
    cap(56,2))``. Serializes ``phase.to_string()`` once and threads it into all
    three helpers (the shared cache key)."""
    pstr = phase.to_string()
    return (
        _location_array(phase, perspective, pstr),
        _used_pile_array(phase, pstr),
        _capability_array(phase, pstr),
    )


def raw_window_arrays(history, perspective, window):
    """Raw arrays for the last ``window`` phases of ``history``, stacked on a
    leading frame axis: ``loc(window,56,9), usp(window,56,22), cap(window,56,2)``.

    Tail-aligned (frame ``window-1`` is the most recent phase), matching the old
    ``BasicNet.tensorify_training`` indexing. Callers pad ``history`` to exactly
    ``window`` (``AlphaZeroNode._trimmed_history``), so head/tail coincide there.
    Nets differ only in how they permute/stack these three arrays.
    """
    loc = np.zeros((window, MAX_CARDS_IN_GAME, MAX_LOCATIONS), dtype=np.float32)
    usp = np.zeros((window, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS), dtype=np.float32)
    cap = np.zeros((window, MAX_CARDS_IN_GAME, CAP_CHANNELS), dtype=np.float32)
    for j in range(window, 0, -1):
        phase = history[-j]
        frame = window - j
        pstr = phase.to_string()  # serialize once; shared cache key for all 3
        loc[frame] = _location_array(phase, perspective, pstr)
        usp[frame] = _used_pile_array(phase, pstr)
        cap[frame] = _capability_array(phase, pstr)
    return loc, usp, cap


# per-card token feature width: one frame contributes location(9) + used_pile(22)
# + capability(2) = 33 features; a card token stacks these over the window frames
FEATURE_WIDTH = MAX_LOCATIONS + int(MAX_PLAYED_STATUS) + CAP_CHANNELS


def fuse_card_tokens(loc, usp, cap):
    """Fuse the three raw window arrays into one token per card. Each of the 56
    cards becomes a row whose features are the window frames'
    ``[location | used_pile | capability]`` concatenated (frame-major within the
    row: frame 0's ``FEATURE_WIDTH`` features, then frame 1's, ...).

    ``loc(window,56,9) usp(window,56,22) cap(window,56,2)`` ->
    ``(56, window*FEATURE_WIDTH)`` float32. This is the shared card-token layout
    for the set-structured nets (CardTransformer / PerCardMLP / Mixer); each wraps
    it with ``torch.from_numpy(...).unsqueeze(0)`` in its ``_assemble``.
    """
    fused = np.concatenate([loc, usp, cap], axis=-1)   # (window, 56, FEATURE_WIDTH)
    fused = np.transpose(fused, (1, 0, 2))             # (56, window, FEATURE_WIDTH)
    return np.ascontiguousarray(fused.reshape(MAX_CARDS_IN_GAME, -1))


def shared_targets(info, cur_phase):
    """The architecture-independent training targets from an ``AZNodeInfo`` and the
    decision phase: ``value`` (scalar z), ``attacking`` (0/1 phase flag),
    ``keepyness`` (56,), ``atk_probs`` (56, 22)."""
    return {
        "value": float(info.value),
        "attacking": float(cur_phase.phase_attacking),
        "keepyness": info.keepyness,
        "atk_probs": info.atk_probs,
    }
