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


def write_json_array(path, items, *, indent=None):
    """Write ``items`` to ``path`` as one clean JSON array (RegiEncoder-encoded).

    The single canonical on-disk log format.  Used for whole-history snapshots
    (WebPlayerLog) and, via :class:`JSONArrayWriter`, for streamed event logs.
    """
    with open(path, "w") as f:
        json.dump(list(items), f, cls=RegiEncoder, indent=indent)


class JSONArrayWriter:
    """Stream objects to a file as one valid JSON array, element by element.

    Produces the same on-disk shape as :func:`write_json_array` (``[a,b,c]``)
    without buffering everything in memory and without the old trailing-``{}``
    sentinel: elements are comma-separated as they arrive and the array is
    closed with ``]`` explicitly.  Use as a context manager or call ``close()``;
    ``__del__`` finalizes as a best-effort fallback.
    """

    def __init__(self, path, *, indent=None):
        self.path = path
        self._indent = indent
        self._fptr = open(path, "w")
        self._fptr.write("[")
        self.count = 0

    def append(self, obj):
        if self._fptr is None:
            raise ValueError("append() on a closed JSONArrayWriter")
        if self.count:
            self._fptr.write(",")
        json.dump(obj, self._fptr, cls=RegiEncoder, indent=self._indent)
        self.count += 1

    def close(self):
        if self._fptr is not None:
            self._fptr.write("]")
            self._fptr.close()
            self._fptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class JSONLog(JSONBaseLog):
    """Streams game events to a file as one clean JSON array.

    Finalizes on ``close()`` / context-manager exit; ``__del__`` is a fallback
    so the legacy ``log = JSONLog(path)`` usage (finalized at GC) still yields
    valid JSON.  An unused log writes ``[]`` rather than the old ``[{}]``.
    """

    def __init__(self, fname):
        super().__init__()
        self.fname = fname
        self.writer = JSONArrayWriter(fname)

    @property
    def count(self):
        return self.writer.count

    def close(self):
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def log(self, obj):
        self.writer.append(obj)
