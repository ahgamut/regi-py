from regi_py.core import *
from regi_py.logging.utils import *
import json
import enum


class RegiEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, GameState):
            return dump_debug(obj)
        elif isinstance(obj, Player):
            return dump_player(obj)
        elif isinstance(obj, Enemy):
            return dump_enemy(obj)
        elif isinstance(obj, BaseStrategy):
            return obj.__strat_name__
        elif isinstance(obj, Combo):
            return dump_combo(obj)
        elif isinstance(obj, Card):
            return dump_card(obj)
        elif isinstance(obj, enum.IntEnum):
            return str(obj.name)
        elif isinstance(obj, enum.Flag):
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
        self.log({"event": GameEvent.STARTGAME.name, "game": dump_debug(game)})

    def endgame(self, reason, game):
        self.log({"event": GameEvent.ENDGAME.name, "game": dump_debug(game)})

    def postgame(self, game):
        self.log({"event": GameEvent.POSTGAME.name, "game": dump_debug(game)})

    ####

    def attack(self, player, enemy, combo, damage, game):
        self.log(
            {
                "event": GameEvent.ATTACK.name,
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
                "event": GameEvent.DEFEND.name,
                "player": player,
                "enemy": game.enemy_pile[0],
                "combo": combo,
                "damage": damage,
                "game": game,
            }
        )

    def redirect(self, player, next_playerid, game):
        self.log(
            {
                "event": GameEvent.REDIRECT.name,
                "player": player,
                "next_playerid": next_playerid,
                "game": game,
            }
        )

    def failBlock(self, player, damage, maxblock, game):
        self.log(
            {
                "event": GameEvent.FAILBLOCK.name,
                "player": player,
                "enemy": game.enemy_pile[0],
                "maxblock": maxblock,
                "damage": damage,
                "game": game,
            }
        )

    def fullBlock(self, player, damage, fullblock, game):
        self.log(
            {
                "event": GameEvent.FULLBLOCK.name,
                "player": player,
                "enemy": game.enemy_pile[0],
                "fullblock": fullblock,
                "damage": damage,
                "game": game,
            }
        )

    def drawOne(self, player):
        self.log({"event": GameEvent.DRAWONE.name, "player": player})

    def cannotDrawDeckEmpty(self, player, game):
        self.log({"event": GameEvent.DECKEMPTY.name, "player": player})

    def replenish(self, n_cards):
        self.log({"event": GameEvent.REPLENISH.name, "n_cards": n_cards})

    def enemyKill(self, enemy, game):
        self.log({"event": GameEvent.ENEMYKILL.name, "enemy": enemy, "game": game})

    ####

    def state(self, game):
        self.log({"event": GameEvent.STATE.name, "game": dump_debug(game)})

    def debug(self, game):
        self.log({"event": GameEvent.DEBUG.name, "game": dump_debug(game)})


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
