"""Backwards-compatible ``dump_*`` helpers.

The dict schema is defined once in :mod:`regi_py.serialize`; this module just
exposes the historical names used by the JSON logs and the dataframe builders.
"""
from regi_py.serialize import (
    CARD_FIELDS,
    ENEMY_FIELDS,
    PLAYER_LIMITED_FIELDS,
    PLAYER_FIELDS,
    GAME_FIELDS,
    DEBUG_EXTRA_FIELDS,
    card_to_dict,
    combo_to_dict,
    enemy_to_dict,
    player_limited_to_dict,
    player_to_dict,
    game_to_dict,
)


def dump_game(game):
    return game_to_dict(game, debug=False)


def dump_debug(game):
    return game_to_dict(game, debug=True)


dump_card = card_to_dict
dump_combo = combo_to_dict
dump_enemy = enemy_to_dict
dump_player_limited = player_limited_to_dict
dump_player = player_to_dict
