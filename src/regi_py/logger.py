from regi_py.core import *
import json

__all__ = ("JSONBaseLog", "JSONLog", "RegiEncoder")


def dump_game(game):
    result = dict()
    result["num_players"] = game.num_players
    result["active_player"] = None
    if game.active_player >= 0 and game.active_player < len(game.players):
        result["active_player"] = dump_player(game.players[game.active_player])
    result["hand_size"] = game.hand_size
    result["players"] = [dump_player_limited(player) for player in game.players]
    result["past_yields"] = game.past_yields
    result["status"] = str(game.status.name)
    result["used_combos"] = [dump_combo(combo) for combo in game.used_combos]
    result["current_enemy"] = None
    result["draw_pile_size"] = len(game.draw_pile)
    result["discard_pile_size"] = len(game.discard_pile)
    result["enemy_pile_size"] = len(game.enemy_pile)
    if len(game.enemy_pile) > 0:
        result["current_enemy"] = dump_enemy(game.enemy_pile[0])
    return result


def dump_debug(game):
    result = dump_game(game)
    result["players"] = [dump_player(player) for player in game.players]
    result["draw_pile"] = [str(x) for x in game.draw_pile]
    result["discard_pile"] = [str(x) for x in game.discard_pile]
    result["enemy_pile"] = [str(x) for x in game.enemy_pile]
    return result


def dump_player_limited(player):
    result = dict()
    result["id"] = player.id
    result["alive"] = player.alive
    result["num_cards"] = len(player.cards)
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


class RegiEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, GameState):
            return dump_game(obj)
        elif isinstance(obj, Player):
            return dump_player(obj)
        elif isinstance(obj, Enemy):
            return dump_enemy(obj)
        elif isinstance(obj, BaseStrategy):
            return obj.__class__.__name__
        elif isinstance(obj, Combo):
            return dump_combo(obj)
        elif isinstance(obj, Card):
            return dump_card(obj)
        elif isinstance(obj, Suit):
            return str(obj.name)
        elif isinstance(obj, Entry):
            return str(obj.name)
        elif isinstance(obj, SuitPower):
            return str(obj.name)
        elif isinstance(obj, EndGameReason):
            return str(obj.name)
        elif isinstance(obj, GameStatus):
            return str(obj.name)
        else:
            return super().default(obj)


class JSONBaseLog(BaseLog):
    def __init__(self):
        super().__init__()

    def log(self, obj):
        raise NotImplemented("subclass this to log objects")

    ####
    def startgame(self, game):
        self.log({"event": "STARTGAME", "game": dump_debug(game)})

    def endgame(self, reason, game):
        self.log({"event": "ENDGAME", "game": dump_game(game)})

    def postgame(self, game):
        self.log({"event": "POSTGAME", "game": dump_debug(game)})

    ####

    def attack(self, player, enemy, combo, damage, game):
        self.log(
            {
                "event": "ATTACK",
                "player": player,
                "enemy": enemy,
                "combo": combo,
                "damage": damage,
                "game": game,
            }
        )

    def defend(self, player, combo, damage, game):
        self.log(
            {
                "event": "DEFEND",
                "player": player,
                "enemy": game.enemy_pile[0],
                "combo": combo,
                "damage": damage,
                "game": game,
            }
        )

    def failBlock(self, player, damage, maxblock, game):
        self.log(
            {
                "event": "FAILBLOCK",
                "player": player,
                "enemy": game.enemy_pile[0],
                "maxblock": maxblock,
                "damage": damage,
                "game": game,
            }
        )

    def drawOne(self, player):
        self.log({"event": "DRAWONE", "player": player})

    def replenish(self, n_cards):
        self.log({"event": "REPLENISH", "n_cards": n_cards})

    def enemyKill(self, enemy, game):
        self.log({"event": "ENEMYKILL", "enemy": enemy, "game": game})

    ####

    def state(self, game):
        self.log({"event": "TURNSTART", "game": dump_game(game)})

    def debug(self, game):
        self.log({"event": "DEBUG", "game": dump_debug(game)})

    def endTurn(self, game):
        self.log({"event": "TURNEND", "game": dump_game(game)})


class JSONLog(JSONBaseLog):
    def __init__(self, fname):
        super().__init__()
        self.fname = fname
        self.fptr = open(fname, "w")
        self.count = 0
        self.fptr.write("[\n")

    def __del__(self):
        if self.fptr:
            self.fptr.write("{}]\n")
            self.fptr.close()
        self.fptr = None

    def log(self, obj):
        json.dump(obj, self.fptr, cls=RegiEncoder)
        self.fptr.write(",\n")
        self.count += 1
