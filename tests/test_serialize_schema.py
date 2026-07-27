"""regi_py.serialize is the single source of truth for game -> dict.

Locks that the schema-field tuples exactly match the keys the serializers emit
(order included) and that the logging dump_* helpers delegate to this module.
"""
import pytest

core = pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

from regi_py.core import GameState, RandomStrategy
from regi_py.logging import DummyLog
from regi_py import serialize
from regi_py.logging import utils as log_utils

from conftest import make_game


# --------------------------------------------------------------------------- #
# exported C++ encoding constants
# --------------------------------------------------------------------------- #
def test_dimension_constants_exported():
    assert core.MAX_CARDS_IN_GAME == 56
    assert core.TOTAL_ENTRY_OPTIONS == 14
    assert core.TOTAL_SUIT_OPTIONS == 4
    # location = entry + TOTAL_ENTRY_OPTIONS * suit spans exactly the card space
    assert core.TOTAL_ENTRY_OPTIONS * core.TOTAL_SUIT_OPTIONS == core.MAX_CARDS_IN_GAME
    # already-present dimensions (via enum export) still resolve
    assert core.MAX_LOCATIONS == 9
    assert core.MAX_PLAYED_STATUS == 22


# --------------------------------------------------------------------------- #
# schema tuples match emitted keys exactly (order included)
# --------------------------------------------------------------------------- #
def _fresh_game(n=2):
    game = GameState(DummyLog())
    for _ in range(n):
        game.add_player(RandomStrategy())
    game.initialize()
    return game


def test_card_and_enemy_schema_matches_keys():
    game = _fresh_game()
    card = game.players[0].cards[0]
    assert tuple(serialize.card_to_dict(card)) == serialize.CARD_FIELDS
    assert tuple(serialize.enemy_to_dict(game.enemy_pile[0])) == serialize.ENEMY_FIELDS


def test_player_schema_matches_keys():
    game = _fresh_game()
    player = game.players[0]
    assert tuple(serialize.player_limited_to_dict(player)) == serialize.PLAYER_LIMITED_FIELDS
    assert tuple(serialize.player_to_dict(player)) == serialize.PLAYER_FIELDS


def test_game_schema_matches_keys_in_order():
    game = _fresh_game()
    assert tuple(serialize.game_to_dict(game, debug=False)) == serialize.GAME_FIELDS
    assert tuple(serialize.game_to_dict(game, debug=True)) == (
        serialize.GAME_FIELDS + serialize.DEBUG_EXTRA_FIELDS
    )


def test_combo_serializes_to_list_of_cards():
    for _ in range(25):
        game = make_game(2)
        game.start_loop()
        for combo in game.used_combos:
            if len(combo.parts) == 0:
                continue
            out = serialize.combo_to_dict(combo)
            assert isinstance(out, list)
            assert all(tuple(c) == serialize.CARD_FIELDS for c in out)
            return
    pytest.skip("no non-empty used combos observed across 25 games")


# --------------------------------------------------------------------------- #
# logging dump_* delegate to the canonical module (same output)
# --------------------------------------------------------------------------- #
def test_logging_dump_helpers_delegate():
    game = _fresh_game(3)
    assert log_utils.dump_game(game) == serialize.game_to_dict(game, debug=False)
    assert log_utils.dump_debug(game) == serialize.game_to_dict(game, debug=True)
    assert log_utils.dump_player(game.players[0]) == serialize.player_to_dict(game.players[0])
    assert log_utils.dump_enemy(game.enemy_pile[0]) == serialize.enemy_to_dict(game.enemy_pile[0])
    card = game.players[0].cards[0]
    assert log_utils.dump_card(card) == serialize.card_to_dict(card)
