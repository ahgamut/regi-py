"""Shared helpers for the RL package.

Kept intentionally small: it re-exports the core and phase-utility namespaces
that ``basicnet`` / ``az_explorer`` pull in via ``from regi_py.rl.utils import *``
(e.g. ``MAX_CARDS_IN_GAME``, ``MAX_PLAYED_STATUS``, ``ComboTable``, ``BaseStrategy``,
``enemy_hp_left``, ``np``), plus the one genuinely shared numeric helper.
"""
from regi_py.core import *
from regi_py.strats.phase_utils import *

import numpy as np


def normalize_probs(arr):
    """Normalize a nonnegative array to sum 1; empty/zero sums throw to the last cell."""
    t = np.sum(arr)
    if t != 0:
        arr /= t
    else:
        # terminal / no-move case: put all mass on the last slot
        arr[-1] = 1.0
    return arr


def hp_loss_penalty(enemy_hp):
    """Shaped reward for a *lost* game, tiered by enemy HP remaining: the closer
    the team came to clearing the board, the smaller the penalty.

    Single source of truth for the loss-side reward shaping shared by the MCTS
    leaf value (``az_explorer.AlphaZeroNode.simulate``) and the whole-game
    training target (``rl.training.run_single_game``). Lives here (torch-free)
    so ``az_explorer`` can use it without pulling in ``training``/torch.
    """
    if enemy_hp > 320:
        return -1.0
    if enemy_hp > 280:
        return -0.9
    return (160 - enemy_hp) / 160
