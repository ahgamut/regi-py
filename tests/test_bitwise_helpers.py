"""Feature helpers: hand/cards bitwise (C++) and the bitwise<->cell map (Python).

The hand bitwise lets a hand (any vector<Card>) be viewed as the same u64
location-bitmask space as Combo.bitwise, so combo-subset checks are one bitwise
op.  The cell map is a fixed bijection between a combo's bitwise identity and its
ComboTable (loc, pst) cell.
"""

import numpy as np
import pytest

import regi_py
from regi_py.core import GameState, RandomStrategy, ComboTable, cards_bitwise
from regi_py.logging import DummyLog
from regi_py.combomap import cell_of_bitwise, bitwise_of_cell, bitwise_to_cell_map

from conftest import make_game
import regi_py.core as core
from regi_py.strats.phase_utils import get_expansion_at


# --------------------------------------------------------------------------- #
# cards / hand -> bitwise
# --------------------------------------------------------------------------- #
def test_hand_bitwise_is_or_of_locations(seeded):
    game = make_game(2)
    for p in game.players:
        expected = 0
        for c in p.cards:
            expected |= 1 << c.location
        assert p.hand_bitwise == expected
        assert cards_bitwise(p.cards) == expected


def test_cards_bitwise_empty_is_zero():
    assert cards_bitwise([]) == 0


def test_offered_combos_are_subsets_of_the_hand(seeded):
    # every combo the engine offers a player is playable from that player's hand,
    # i.e. its bitwise is a subset of the hand bitmask
    checked = 0
    for _ in range(30):
        game = make_game(2)
        phase = game.export_phaseinfo()
        next_phases, combos = get_expansion_at(phase)
        if not combos:
            continue
        actor = game.players[game.active_player]
        hand = actor.hand_bitwise
        for c in combos:
            # yield (bitwise 0) is trivially a subset; real combos must fit
            assert (c.bitwise & ~hand) == 0
        checked += 1
        if checked >= 3:
            break
    assert checked > 0, "expected at least one decision with offered combos"


# --------------------------------------------------------------------------- #
# bitwise <-> ComboTable (loc, pst)
# --------------------------------------------------------------------------- #
def test_map_is_a_bijection():
    mapping = bitwise_to_cell_map()
    table = ComboTable.all_entries()
    n_cells = int(np.asarray(table).sum())
    # one bitwise key per viable cell (yield's phantom bitwise-1 collapses onto 0,
    # so key count still equals cell count), and cells are unique
    assert len(mapping) == n_cells
    assert len(set(mapping.values())) == n_cells


def test_cell_roundtrip_for_every_cell():
    for bitwise, (loc, pst) in bitwise_to_cell_map().items():
        assert bitwise_of_cell(loc, pst) == bitwise
        assert cell_of_bitwise(bitwise) == (loc, pst)


def test_yield_is_normalized_to_zero():
    assert cell_of_bitwise(0) == (0, 0)
    assert bitwise_of_cell(0, 0) == 0
    # the phantom yield-card bitwise (1) is never a key
    assert 1 not in bitwise_to_cell_map()


def test_cell_of_bitwise_missing_returns_default():
    absent = 1 << 63  # no location reaches bit 63
    assert cell_of_bitwise(absent) is None
    assert cell_of_bitwise(absent, default=(-1, -1)) == (-1, -1)


def test_map_matches_a_built_combotable(seeded):
    # a combo resolved through the precomputed map lands on the same cell the
    # engine's ComboTable would mark for it
    for _ in range(30):
        phase = make_game(2).export_phaseinfo()
        _, combos = get_expansion_at(phase)
        real = [c for c in combos if c.bitwise != 0]
        if real:
            break
    assert real, "expected at least one non-yield combo"
    for c in real:
        t = ComboTable.empty()
        t.add_used_pile([c])
        loc, pst = (int(x[0]) for x in np.asarray(t).nonzero())
        assert cell_of_bitwise(c.bitwise) == (loc, pst)
