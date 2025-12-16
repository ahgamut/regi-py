from regi_py.core import *
from regi_py.logging.utils import *
import json


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
        self.log({"event": "STATE", "game": dump_game(game)})

    def debug(self, game):
        self.log({"event": "DEBUG", "game": dump_debug(game)})

    def startPlayerTurn(self, game):
        self.log({"event": "TURNSTART", "game": dump_debug(game)})

    def endPlayerTurn(self, game):
        self.log({"event": "TURNEND", "game": dump_debug(game)})


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
