"""Shared building blocks for the regi webapp servers.

``driver.py`` (multiplayer, CLI-configured) and ``pubdrive.py`` (per-user SQLite
sessions, env-configured) both import from here. Everything shared but
session-model-agnostic lives here; each server keeps only its own
``ConnectionManager``/``Context``/routes/game-loop.

Nothing webapp-specific lives in ``src/`` -- this module is the home for it. The
one dependency into ``src`` is the general serialization/logging in
``regi_py.logging`` (``RegiEncoder``/``JSONBaseLog``/``write_json_array``), which
also drives on-disk logs and is NOT webapp-specific.
"""
import datetime
import json
import logging
import os
import sys
import traceback

import anyio
from anyio.from_thread import BlockingPortalProvider
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from regi_py.strats import BaseStrategy
from regi_py import RegiEncoder, JSONBaseLog, write_json_array

logger = logging.getLogger("regi")

_HERE = os.path.dirname(__file__)


class GameInterruptedError(Exception):
    """Raised when the WebSocket disconnects mid-game to unwind the game thread."""


# --------------------------------------------------------------------------- #
# wire: single-encoded JSON
# --------------------------------------------------------------------------- #
async def send_encoded(ws, message):
    """Serialize ``message`` once (RegiEncoder) and send it as a WS text frame.

    The single source of the wire format. ``websocket.send_json`` would JSON-encode
    a second time (the old double-encode the frontend compensated for with
    ``JSON.parse(JSON.parse(...))``); ``send_text`` of a pre-encoded string keeps it
    single so the client does one ``JSON.parse``.
    """
    await ws.send_text(json.dumps(message, cls=RegiEncoder))


# --------------------------------------------------------------------------- #
# username enrichment (parameterized by a per-server name resolver)
# --------------------------------------------------------------------------- #
def enrich_player(player_dict, resolve_name):
    if isinstance(player_dict, dict) and "id" in player_dict:
        name = resolve_name(player_dict["id"])
        if name:
            player_dict["username"] = name


def enrich_with_usernames(data, resolve_name):
    """Walk ``data`` and stamp ``username`` onto any player dicts, using
    ``resolve_name(player_id) -> Optional[str]`` (each server supplies its own:
    multiplayer maps over its user list, single-user returns its one name)."""
    if isinstance(data, dict):
        for key in ("player", "active_player"):
            if isinstance(data.get(key), dict):
                enrich_player(data[key], resolve_name)
        if isinstance(data.get("players"), list):
            for p in data["players"]:
                enrich_player(p, resolve_name)
        for key, value in data.items():
            if key not in ("player", "active_player", "players"):
                data[key] = enrich_with_usernames(value, resolve_name)
    elif isinstance(data, list):
        data = [enrich_with_usernames(item, resolve_name) for item in data]
    return data


# --------------------------------------------------------------------------- #
# lighter wire payload
# --------------------------------------------------------------------------- #
# Every log event that mutates the board is immediately followed by a STATE event
# carrying the full game snapshot (verified against real logs), and the frontend
# handlers for these events read only their own fields (combo/enemy/damage/...),
# not `game`. So the full `game` dict on them is redundant on the wire -- keep it
# only on the events that are the board's actual source of truth.
_WIRE_KEEP_GAME = frozenset(
    {"STARTGAME", "STATE", "ENDGAME", "POSTGAME", "REDIRECT", "DEBUG"}
)


def trim_for_wire(obj):
    """Return a wire-lean copy of a log event: drop the redundant full ``game``
    snapshot from board-mutating events (a STATE with ``game`` always follows).
    Returns the object unchanged when there's nothing to trim; otherwise a shallow
    copy, so the caller's stored history keeps the full object."""
    if not isinstance(obj, dict):
        return obj
    if obj.get("event") in _WIRE_KEEP_GAME or "game" not in obj:
        return obj
    trimmed = dict(obj)
    trimmed.pop("game", None)
    return trimmed


# --------------------------------------------------------------------------- #
# the human player's strategy (drives the websocket round-trips)
# --------------------------------------------------------------------------- #
class WebPlayerStrategy(BaseStrategy):
    """A human player: each decision is sent to the browser and the game thread
    blocks on the websocket response. Session-agnostic -- it talks to ``ctx.manager``
    and resolves names via ``ctx.resolve_name``. An optional ``recommender`` (any
    ``RecommenderMixin``) attaches suggested moves to attack/defense prompts."""

    __strat_name__ = "player-webui"

    def __init__(self, userid, username, websocket, ctx, recommender=None):
        super().__init__()
        self.__strat_name__ = f"player-webui-{username}"
        self.userid = userid
        self.username = username
        self.websocket = websocket
        self._ctx = ctx
        self.recommender = recommender
        self.portal_provider = BlockingPortalProvider()
        self.response = None
        self.ready = False
        self.disconnected = False

    @staticmethod
    async def comms_twoway(self, websocket, obj):
        # already-dropped player: unwind the game thread before sending a prompt
        # to a socket that is gone (no resume -- the game is ending).
        if self.disconnected:
            raise GameInterruptedError("player disconnected")
        enrich_with_usernames(obj, self._ctx.resolve_name)
        logger.debug("sending %s", obj.get("type"))
        try:
            await self._ctx.manager.send_dict(obj, websocket)
        except Exception as exc:
            raise GameInterruptedError("WS send failed") from exc

        while self.response is None:
            if self.disconnected:
                raise GameInterruptedError("player disconnected")
            await anyio.sleep(0.5)

        resp = self.response
        self.response = None
        return resp

    def _ask(self, result):
        with self.portal_provider as portal:
            return portal.call(
                WebPlayerStrategy.comms_twoway, self, self.websocket, result
            )

    @staticmethod
    async def _send_assign(manager, websocket, playerid):
        try:
            await manager.send_dict({"type": "assign-id", "playerid": playerid}, websocket)
        except Exception:
            pass  # a failed seat notice must not crash game startup

    def notify_playerid(self, playerid):
        """Tell this player's client which seat (game player id) it drew this
        game, so the frontend tracks 'my turn'/'my cards' after a seat shuffle."""
        with self.portal_provider as portal:
            portal.call(
                WebPlayerStrategy._send_assign, self._ctx.manager, self.websocket, playerid
            )

    def _reco(self, combos, game):
        if self.recommender is None:
            return None
        return self.recommender.getRecommendedMoves(game.export_phaseinfo(), combos)

    def setup(self, player, game):
        if self.ready:
            return 0
        response = self._ask({"type": "ready", "player": player, "game": game})
        option = int(response.get("choice", -1))
        return option if option >= 0 else -1

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        data = {
            "userid": self.userid,
            "player": player,
            "combos": combos,
            "yield_allowed": yield_allowed,
            "game": game,
        }
        reco = self._reco(combos, game)
        if reco is not None:
            data["reco"] = reco
        response = self._ask({"type": "select-attack", "data": data})
        option = int(response.get("choice", -1))
        if option < 0 or option > len(combos):
            option = -1
        return option

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        data = {
            "userid": self.userid,
            "player": player,
            "combos": combos,
            "damage": damage,
            "game": game,
        }
        reco = self._reco(combos, game)
        if reco is not None:
            data["reco"] = reco
        response = self._ask({"type": "select-defend", "data": data})
        option = int(response.get("choice", -1))
        if option < 0 or option > len(combos):
            option = -1
        return option

    def getRedirectIndex(self, player, game):
        data = {"userid": self.userid, "player": player, "game": game}
        response = self._ask({"type": "select-redirect", "data": data})
        option = int(response.get("choice", -1))
        if option < 0 or option > game.num_players or option == player.id:
            option = -1
        return option


# --------------------------------------------------------------------------- #
# game log -> websocket broadcast (+ full history for save/download)
# --------------------------------------------------------------------------- #
class WebPlayerLog(JSONBaseLog):
    """Broadcasts each game event to the client (trimmed for the wire, enriched
    with usernames) while keeping the FULL events in ``self.history`` for the
    optional per-game JSON save (``history_folder``)."""

    def __init__(self, manager, resolve_name, history_folder=None):
        super().__init__()
        self.manager = manager
        self.resolve_name = resolve_name
        self.history = []
        self.history_folder = history_folder
        self.portal_provider = BlockingPortalProvider()
        self.count = 0

    def startgame(self, game):
        self.history.clear()
        super().startgame(game)

    def postgame(self, game):
        super().postgame(game)
        if self.history_folder:
            os.makedirs(self.history_folder, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.history_folder, f"game_{timestamp}.json")
            write_json_array(filepath, self.history, indent=2)

    @staticmethod
    async def log_actual(manager, obj, resolve_name):
        wire = enrich_with_usernames(trim_for_wire(obj), resolve_name)
        await manager.broadcast_dict({"type": "log", "data": wire})

    def log(self, obj):
        self.history.append(obj)  # full object retained for save/download
        with self.portal_provider as portal:
            portal.call(WebPlayerLog.log_actual, self.manager, obj, self.resolve_name)
        self.count += 1


# --------------------------------------------------------------------------- #
# FastAPI app scaffolding
# --------------------------------------------------------------------------- #
async def catchall_exception_handler(request: Request, exc: Exception):
    _, _, exc_traceback = sys.exc_info()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(
            {"error": True, "traceback": "".join(traceback.format_tb(exc_traceback))}
        ),
    )


def make_app(lifespan=None):
    """Build a FastAPI app with /static mounted, Jinja2 templates, permissive CORS,
    and the catch-all exception handler. Returns ``(app, templates)``."""
    app = FastAPI(docs_url=None, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")))
    templates = Jinja2Templates(directory=os.path.join(_HERE, "templates"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(Exception, catchall_exception_handler)
    return app, templates


# --------------------------------------------------------------------------- #
# recommender spec: "NAME-ITERS" (+ optional weights for NN nets)
# --------------------------------------------------------------------------- #
BUILTIN_RECOS = ("brute", "mcts")


def parse_reco_spec(spec):
    """``"NAME-ITERS"`` -> ``(name, iters)``. Raises ``ValueError`` on bad grammar.
    Net names have no ``-``, so split on the last one."""
    name, sep, iters_s = spec.rpartition("-")
    if not sep or not name:
        raise ValueError(
            f"--reco must be NAME-ITERS (e.g. brute-128, mcts-64, adzmulti-0); got {spec!r}"
        )
    try:
        iters = int(iters_s)
    except ValueError:
        raise ValueError(f"--reco ITERS must be an integer; got {iters_s!r}")
    if iters < 0:
        raise ValueError("--reco ITERS must be >= 0")
    return name, iters


def validate_reco(spec, weights_path):
    """Torch-free startup validation: parse the spec and require a weights file for
    NN names. Returns ``(name, iters)``. Does NOT import torch or resolve the net
    (that happens lazily in ``make_recommender``). Raises ``ValueError`` on error."""
    name, iters = parse_reco_spec(spec)
    if name not in BUILTIN_RECOS:
        if not weights_path:
            raise ValueError(
                f"NN recommender {name!r} requires --reco-weights PATH"
            )
        if not os.path.isfile(weights_path):
            raise ValueError(f"--reco-weights file not found: {weights_path!r}")
    return name, iters


def make_recommender(name, iters, weights_path=None):
    """Construct a recommender strategy. brute/mcts import no torch. Any other name
    is an NN net: torch + the net registries are imported LAZILY here (so a
    brute/mcts default never pulls torch). ``iters == 0`` -> search-free
    Direct-net; ``iters > 0`` -> Explorer(iterations). AZ vs ADZ is resolved by
    which registry holds the name."""
    if name == "brute":
        from regi_py.strats import BruteSamplingStrategy

        return BruteSamplingStrategy(iters)
    if name == "mcts":
        from regi_py.strats.mcts_explorer import MCTSExplorerStrategy

        return MCTSExplorerStrategy(iters)
    return _make_nn_recommender(name, iters, weights_path)


def _make_nn_recommender(name, iters, weights_path):
    if not weights_path:
        raise ValueError(f"NN recommender {name!r} requires --reco-weights PATH")
    # NN-net construction lives in src (``regi_py.rl.make_net_strategy``) -- it's
    # an rl concern, not webapp-specific -- so bots and recommenders share it.
    # torch + the net registries are imported lazily there.
    from regi_py.rl import make_net_strategy

    return make_net_strategy(name, iters, weights_path)
