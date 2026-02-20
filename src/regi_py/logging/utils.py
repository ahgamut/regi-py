from regi_py.core import *


def dump_game(game):
    result = dict()
    result["num_players"] = game.num_players
    result["active_player_id"] = None
    result["active_player"] = None
    if game.active_player >= 0 and game.active_player < len(game.players):
        result["active_player_id"] = game.active_player
        result["active_player"] = dump_player(game.players[game.active_player])
    result["phase_count"] = game.phase_count
    result["phase_attacking"] = game.phase_attacking
    result["hand_size"] = game.hand_size
    result["players"] = [dump_player(player) for player in game.players]
    result["past_yields"] = game.past_yields
    result["status"] = str(game.status.name)
    result["used_combos"] = [dump_combo(combo) for combo in game.used_combos if len(combo.parts) != 0]
    result["current_enemy"] = None
    result["draw_pile_size"] = len(game.draw_pile)
    result["discard_pile_size"] = len(game.discard_pile)
    result["enemy_pile_size"] = len(game.enemy_pile)
    if len(game.enemy_pile) > 0:
        result["current_enemy"] = dump_enemy(game.enemy_pile[0])
    result["enemy_pile"] = [str(x) for x in game.enemy_pile]
    return result


def dump_debug(game):
    result = dump_game(game)
    result["players"] = [dump_player(player) for player in game.players]
    result["draw_pile"] = [str(x) for x in game.draw_pile]
    result["discard_pile"] = [str(x) for x in game.discard_pile]
    return result


def dump_player_limited(player):
    result = dict()
    result["id"] = player.id
    result["alive"] = player.alive
    result["num_cards"] = len(player.cards)
    result["strategy"] = player.strategy
    return result


def dump_player(player):
    result = dump_player_limited(player)
    result["cards"] = [str(card) for card in player.cards]
    return result


def dump_enemy(enemy):
    result = dict()
    result["value"] = str(enemy)
    result["hp"] = enemy.hp
    result["strength"] = enemy.strength
    return result


def dump_card(card):
    result = dict()
    result["value"] = str(card)
    result["strength"] = card.strength
    return result


def dump_combo(combo):
    result = [dump_card(card) for card in combo.parts]
    return result
