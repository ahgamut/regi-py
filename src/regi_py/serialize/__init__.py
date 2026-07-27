"""Single source of truth for turning C++ game objects into plain dicts.

Every ``game -> dict`` serialization surface (the JSON event logs in
``regi_py.logging``, the dataframe builders in ``game_json``) is defined here
exactly once, so the dict schema lives in a single place.  ``regi_py.logging``
re-exports these as the historical ``dump_*`` helpers.

Card ``value`` fields are human-readable display strings; the canonical machine
encoding of a card is its integer ``location`` (see ``Card.location``) and of a
combo its ``bitwise`` (a u64 bitmask over locations).
"""
from regi_py.core import GameState, Player, Enemy, Combo, Card  # noqa: F401

# --------------------------------------------------------------------------- #
# schema: the exact dict keys emitted for each object, in emission order.
# Downstream code (dataframe column builders, tests) derives from these tuples
# instead of re-hardcoding the field names.
# --------------------------------------------------------------------------- #
CARD_FIELDS = ("value", "strength")
ENEMY_FIELDS = ("value", "hp", "strength")
PLAYER_LIMITED_FIELDS = ("id", "alive", "num_cards", "strategy")
PLAYER_FIELDS = PLAYER_LIMITED_FIELDS + ("cards",)
GAME_FIELDS = (
    "num_players",
    "active_player_id",
    "active_player",
    "phase_count",
    "phase_attacking",
    "hand_size",
    "players",
    "past_yields",
    "status",
    "used_combos",
    "current_enemy",
    "current_block",
    "progress",
    "draw_pile_size",
    "discard_pile_size",
    "enemy_pile_size",
    "enemy_pile",
)
# extra keys present only in the debug (full-state) view
DEBUG_EXTRA_FIELDS = ("draw_pile", "discard_pile")


def card_to_dict(card):
    return {"value": str(card), "strength": card.strength}


def combo_to_dict(combo):
    return [card_to_dict(card) for card in combo.parts]


def enemy_to_dict(enemy):
    return {"value": str(enemy), "hp": enemy.hp, "strength": enemy.strength}


def player_limited_to_dict(player):
    return {
        "id": player.id,
        "alive": player.alive,
        "num_cards": len(player.cards),
        "strategy": player.strategy,
    }


def player_to_dict(player):
    result = player_limited_to_dict(player)
    result["cards"] = [str(card) for card in player.cards]
    return result


def game_to_dict(game, debug=False):
    """Serialize a GameState to a dict.

    ``debug=True`` adds the full draw/discard pile contents (the historical
    ``dump_debug`` view); otherwise only pile sizes are emitted.
    """
    result = dict()
    result["num_players"] = game.num_players
    result["active_player_id"] = None
    result["active_player"] = None
    if 0 <= game.active_player < len(game.players):
        result["active_player_id"] = game.active_player
        result["active_player"] = player_to_dict(game.players[game.active_player])
    result["phase_count"] = game.phase_count
    result["phase_attacking"] = game.phase_attacking
    result["hand_size"] = game.hand_size
    result["players"] = [player_to_dict(player) for player in game.players]
    result["past_yields"] = game.past_yields
    result["status"] = str(game.status.name)
    result["used_combos"] = [
        combo_to_dict(combo) for combo in game.used_combos if len(combo.parts) != 0
    ]
    result["current_enemy"] = None
    result["current_block"] = 0
    result["progress"] = 0
    result["draw_pile_size"] = len(game.draw_pile)
    result["discard_pile_size"] = len(game.discard_pile)
    result["enemy_pile_size"] = len(game.enemy_pile)
    if len(game.enemy_pile) > 0:
        result["current_enemy"] = enemy_to_dict(game.enemy_pile[0])
        result["current_block"] = game.get_current_block(game.enemy_pile[0])
        result["progress"] = 360 - sum(e.hp for e in game.enemy_pile if e.hp > 0)
    else:
        result["progress"] = 360
    result["enemy_pile"] = [str(x) for x in game.enemy_pile]
    if debug:
        result["draw_pile"] = [str(x) for x in game.draw_pile]
        result["discard_pile"] = [str(x) for x in game.discard_pile]
    return result
