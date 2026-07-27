"""Characterization tests for every serialization surface of the C++ core.

Covers the compact PhaseInfo string format, pickle (which is string-based),
GameState string/phaseinfo export + restore, and the LocationInfo / ComboTable
numpy buffers. These guard the serialization-unification refactors.
"""

import pickle

import numpy as np
import pytest

from regi_py.core import (
    GameState,
    RandomStrategy,
    PhaseInfo,
    LocationInfo,
    ComboTable,
)
from regi_py.logging import DummyLog

from conftest import make_game


def _combo_key(combo):
    # bitwise is the canonical location-based combo identity (u64 bitmask)
    return combo.bitwise


def _phases_from_played_games(num_games=8, num_players=2):
    """Yield PhaseInfo snapshots taken from the history of completed games."""
    for _ in range(num_games):
        game = make_game(num_players)
        game.start_loop()
        for phase in game.history:
            yield phase


# --------------------------------------------------------------------------- #
# PhaseInfo compact string
# --------------------------------------------------------------------------- #
def test_export_string_equals_phaseinfo_string():
    game = make_game(3)
    assert game.export_string() == game.export_phaseinfo().to_string()


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_string_roundtrip_on_fresh_states(num_players):
    for _ in range(10):
        game = make_game(num_players)
        s = game.export_string()
        assert PhaseInfo.from_string(s).to_string() == s


def test_string_roundtrip_on_history():
    count = 0
    for phase in _phases_from_played_games():
        s = phase.to_string()
        assert PhaseInfo.from_string(s).to_string() == s
        count += 1
    assert count > 0, "expected history phases to test"


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_init_string_restores_state(num_players):
    # export_string() -> _init_string() into a fresh game is exactly idempotent
    game = make_game(num_players)
    s = game.export_string()
    info = PhaseInfo.from_string(s)
    assert info.num_players == num_players

    restored = GameState(DummyLog())
    for _ in range(info.num_players):
        restored.add_player(RandomStrategy())
    restored._init_string(s)
    assert restored.export_string() == s


def test_pickle_is_string_equivalent():
    game = make_game(2)
    info = game.export_phaseinfo()
    loaded = pickle.loads(pickle.dumps(info))
    assert loaded.to_string() == info.to_string()
    assert hash(loaded) == hash(info)


@pytest.mark.parametrize(
    "bad",
    ["", "garbage", "0#1#2#0#0", "9#9#9#9#9!!!!!!", "not-a-phase!!!!", "0#1#5#0#0!"],
)
def test_malformed_string_raises(bad):
    with pytest.raises(Exception):
        PhaseInfo.from_string(bad)


# --------------------------------------------------------------------------- #
# LocationInfo buffer
# --------------------------------------------------------------------------- #
def test_locationinfo_buffer_shape_and_readonly():
    info = make_game(2).export_phaseinfo()
    arr = np.asarray(LocationInfo.from_phase(info))
    assert arr.shape == (56, 9)  # MAX_CARDS_IN_GAME x MAX_LOCATIONS
    assert arr.dtype == np.uint32
    assert arr.flags.writeable is False


def test_locationinfo_each_card_in_exactly_one_location():
    # every card row sums to exactly 1 (a card is in exactly one location,
    # including the NOT_IN_GAME column)
    for info in (make_game(2).export_phaseinfo(), make_game(4).export_phaseinfo()):
        arr = np.asarray(LocationInfo.from_phase(info))
        assert np.all(arr.sum(axis=1) == 1)


# --------------------------------------------------------------------------- #
# ComboTable buffer + used-pile round-trip
# --------------------------------------------------------------------------- #
def test_combotable_buffer_shape_and_readonly():
    info = make_game(2).export_phaseinfo()
    arr = np.asarray(ComboTable.from_phase(info))
    assert arr.shape == (56, 22)  # MAX_CARDS_IN_GAME x MAX_PLAYED_STATUS
    assert arr.dtype == np.uint32
    assert arr.flags.writeable is False


def test_combotable_empty_roundtrip():
    table = ComboTable.empty()
    table.add_used_pile([])
    assert list(table.as_used_pile()) == []
    assert int(np.asarray(table).sum()) == 0


def test_combotable_used_pile_roundtrip():
    examined = 0
    for phase in _phases_from_played_games(num_games=30):
        used = list(phase.used_combos)
        if not used:
            continue
        table = ComboTable.empty()
        table.add_used_pile(used)
        back = list(table.as_used_pile())
        assert sorted(map(_combo_key, used)) == sorted(map(_combo_key, back))
        examined += 1
        if examined >= 5:
            break
    assert examined > 0, "expected some phases with used combos"


def test_combotable_make_combo_matches_nonzero_cells():
    for phase in _phases_from_played_games(num_games=30):
        used = list(phase.used_combos)
        if not used:
            continue
        table = ComboTable.empty()
        table.add_used_pile(used)
        arr = np.asarray(table)
        rows, cols = arr.nonzero()
        # make_combo yields exactly one combo per nonzero cell (some cells, e.g.
        # yield/joker entries, legitimately decode to an empty-parts combo).
        rebuilt = [ComboTable.make_combo(int(r), int(c)) for r, c in zip(rows, cols)]
        assert len(rebuilt) == int(arr.sum())
        return
    pytest.skip("no phases with used combos found")
