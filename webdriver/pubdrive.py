import contextlib
import datetime
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import traceback
from typing import Optional
from uuid import uuid4

logger = logging.getLogger("regi")

import anyio
from anyio.from_thread import BlockingPortalProvider
from fastapi import Cookie, Form
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi import Request
from fastapi import status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

###
from regi_py.strats import BaseStrategy
from regi_py import get_strategy_map
from regi_py import RegiEncoder, JSONBaseLog, GameState, GameStatus


###
class GameInterruptedError(Exception):
    """Raised when the WebSocket disconnects mid-game to unwind the game thread."""
    pass


class SessionStore:
    """SQLite-backed session store for cross-worker session sharing."""

    def __init__(self, session_dir):
        os.makedirs(session_dir, exist_ok=True)
        self.db_path = os.path.join(session_dir, "regi_sessions.db")
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    userid TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    bots TEXT NOT NULL DEFAULT '[]',
                    phase TEXT NOT NULL DEFAULT 'bot_select',
                    created_at REAL NOT NULL
                )
            """)

    def create(self, userid, username):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (userid, username, bots, phase, created_at) VALUES (?, ?, '[]', 'bot_select', ?)",
                (userid, username, time.time()),
            )

    def load(self, userid) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT userid, username, bots, phase FROM sessions WHERE userid = ?",
                (userid,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def set_bots(self, userid, bots):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET bots = ?, phase = 'playing' WHERE userid = ?",
                (json.dumps(bots), userid),
            )

    def set_phase(self, userid, phase):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET phase = ? WHERE userid = ?",
                (phase, userid),
            )

    def delete(self, userid):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE userid = ?", (userid,))


###
class ConnectionManager:
    def __init__(self):
        self.websocket: Optional[WebSocket] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.websocket = websocket

    async def disconnect(self):
        if self.websocket is not None:
            await self.websocket.close(code=1000, reason=None)
            self.websocket = None

    async def send_dict(self, message: dict, websocket: Optional[WebSocket] = None):
        ws = websocket or self.websocket
        if ws is not None:
            raw = json.dumps(message, cls=RegiEncoder)
            await ws.send_json(raw)

    async def broadcast_dict(self, message: dict):
        await self.send_dict(message)


class WebPlayerStrategy(BaseStrategy):
    __strat_name__ = "player-webui"

    def __init__(self, userid, username, websocket, ctx):
        super().__init__()
        self.userid = userid
        self.username = username
        self.websocket = websocket
        self._ctx = ctx
        self.portal_provider = BlockingPortalProvider()
        self.response = None
        self.ready = False
        self.disconnected = False

    @staticmethod
    async def comms_twoway(self, websocket, obj):
        enrich_with_usernames(obj, self._ctx)
        logger.debug("sending %s", obj)
        try:
            await self._ctx.manager.send_dict(obj, websocket)
        except Exception:
            raise GameInterruptedError("WS send failed")

        while self.response is None:
            if self.disconnected:
                raise GameInterruptedError("player disconnected")
            await anyio.sleep(0.5)

        resp = self.response
        self.response = None
        return resp

    def setup(self, player, game):
        if self.ready:
            return 0
        msg = {"type": "ready", "player": player, "game": game}
        with self.portal_provider as portal:
            response = portal.call(
                WebPlayerStrategy.comms_twoway, self, self.websocket, msg
            )
        option = int(response.get("choice", -1))
        if option < 0:
            option = -1
        return option

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
        result = {"type": "select-attack", "data": data}

        with self.portal_provider as portal:
            response = portal.call(
                WebPlayerStrategy.comms_twoway, self, self.websocket, result
            )

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
        result = {"type": "select-defend", "data": data}

        with self.portal_provider as portal:
            response = portal.call(
                WebPlayerStrategy.comms_twoway, self, self.websocket, result
            )

        option = int(response.get("choice", -1))
        if option < 0 or option > len(combos):
            option = -1
        return option

    def getRedirectIndex(self, player, game):
        data = {
            "userid": self.userid,
            "player": player,
            "game": game,
        }
        result = {"type": "select-redirect", "data": data}

        with self.portal_provider as portal:
            response = portal.call(
                WebPlayerStrategy.comms_twoway, self, self.websocket, result
            )

        option = int(response.get("choice", -1))
        if option < 0 or option > game.num_players or option == player.id:
            option = -1
        return option


def enrich_player(player_dict, ctx):
    if isinstance(player_dict, dict) and "id" in player_dict:
        if player_dict["id"] == 0 and ctx.username:
            player_dict["username"] = ctx.username


def enrich_with_usernames(data, ctx):
    if isinstance(data, dict):
        if "player" in data and isinstance(data["player"], dict):
            enrich_player(data["player"], ctx)
        if "active_player" in data and isinstance(data["active_player"], dict):
            enrich_player(data["active_player"], ctx)
        if "players" in data and isinstance(data["players"], list):
            for p in data["players"]:
                enrich_player(p, ctx)
        for key, value in data.items():
            if key not in ("player", "active_player", "players"):
                data[key] = enrich_with_usernames(value, ctx)
    elif isinstance(data, list):
        data = [enrich_with_usernames(item, ctx) for item in data]
    return data


class WebPlayerLog(JSONBaseLog):
    def __init__(self, manager, ctx=None, history_folder=None):
        super().__init__()
        self.manager = manager
        self._ctx = ctx
        self.count = 0
        self.portal_provider = BlockingPortalProvider()
        self.history = []
        self.history_folder = history_folder

    def startgame(self, game):
        self.history.clear()
        super().startgame(game)

    def postgame(self, game):
        super().postgame(game)
        if self.history_folder:
            os.makedirs(self.history_folder, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.history_folder, f"game_{timestamp}.json")
            with open(filepath, "w") as f:
                json.dump(self.history, f, cls=RegiEncoder, indent=2)

    @staticmethod
    async def log_actual(manager, obj, ctx):
        result = {}
        result["type"] = "log"

        enriched_obj = enrich_with_usernames(obj, ctx)
        result["data"] = enriched_obj
        await manager.broadcast_dict(result)

    def log(self, obj):
        self.history.append(obj)
        with self.portal_provider as portal:
            portal.call(WebPlayerLog.log_actual, self.manager, obj, self._ctx)
        self.count += 1


###
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config = {
        "password": os.environ.get("REGI_PASSWORD", "regi"),
        "no_download": os.environ.get("REGI_NO_DOWNLOAD", "").lower() in ("1", "true", "yes"),
        "history_folder": os.environ.get("REGI_HISTORY_FOLDER", None),
    }
    session_dir = os.environ.get("REGI_SESSION_DIR", "/tmp")
    app.state.session_store = SessionStore(session_dir)
    app.state.active_games = {}
    app.state.bot_options = list(get_strategy_map(rl_mods=False).keys())
    logger.info("Regi webserver started")
    yield


app = FastAPI(docs_url=None, lifespan=lifespan)
app.mount(
    "/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"))
)
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def catchall_exception_handler(request: Request, exc: Exception):
    _, _, exc_traceback = sys.exc_info()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(
            {
                "error": True,
                "traceback": "".join(traceback.format_tb(exc_traceback)),
            }
        ),
    )


###


class Context:
    def __init__(self, userid, username, bots, history_folder=None):
        self.manager = ConnectionManager()
        self.playerlog = WebPlayerLog(self.manager, ctx=self, history_folder=history_folder)
        self.game = GameState(self.playerlog)
        self.strats = []
        self.bots = bots
        self.userid = userid
        self.username = username
        self.ALT_STARTED = False
        self.GLOB_THREAD = None
        self.disconnected = False

    @property
    def needs_bot_selection(self):
        return len(self.bots) == 0

    def set_bots(self, bots):
        self.bots = bots

    def load_game(self):
        assert self.ALT_STARTED
        strategy_map = get_strategy_map(rl_mods=False)
        if len(self.strats) != 1 + len(self.bots):
            # first game: add human player + create bot strats
            self.game.add_player(self.strats[0])
            for b in self.bots:
                strat = strategy_map[b]()
                self.strats.append(strat)
                self.game.add_player(self.strats[-1])
        else:
            # reset game: re-add existing strats
            for s in self.strats:
                self.game.add_player(s)

        logger.info("starting with %s", [x.__strat_name__ for x in self.strats])
        assert len(self.strats) >= 2
        assert len(self.strats) <= 4
        self.strats[0].ready = True
        self.game.initialize()
        self.game.start_loop()

    def end_game(self):
        assert self.game is not None
        del self.game
        self.game = None

    def reset_game(self):
        self.game = GameState(self.playerlog)


def per_user_game_loop(ctx):
    """Game loop for a single user's session. Runs in a dedicated thread."""
    ctx.ALT_STARTED = True
    while not ctx.disconnected:
        while ctx.needs_bot_selection and not ctx.disconnected:
            logger.debug("waiting for bot selection for %s", ctx.userid)
            time.sleep(1)
        if ctx.disconnected:
            break

        while len(ctx.strats) < 1 and not ctx.disconnected:
            logger.debug("waiting for player %s", ctx.userid)
            time.sleep(1)
        if ctx.disconnected:
            break

        try:
            ctx.load_game()
        except GameInterruptedError:
            logger.info("game interrupted for %s", ctx.userid)
            break
        except Exception:
            logger.exception("game error for %s", ctx.userid)
            break

        ctx.end_game()

        # wait for reset or disconnect
        while ctx.game is None and not ctx.disconnected:
            time.sleep(1)

    logger.info("game thread exiting for %s", ctx.userid)


def player_join(ctx, websocket):
    if len(ctx.strats) >= 1:
        return
    strat = WebPlayerStrategy(ctx.userid, ctx.username, websocket, ctx)
    ctx.strats.append(strat)


###


# pylint: disable=W0613
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("pages/login.html", {"request": request})


@app.post("/login", response_class=RedirectResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if password != app.state.config["password"]:
        return templates.TemplateResponse(
            "pages/login.html", {"request": request, "error": "Incorrect password."}
        )
    userid = str(uuid4())
    app.state.session_store.create(userid, username)
    response = RedirectResponse(url="/select-bots", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="userid", value=userid)
    response.set_cookie(key="username", value=username)
    return response


@app.get("/select-bots", response_class=HTMLResponse)
def select_bots_page(
    request: Request,
    userid: Optional[str] = Cookie(None),
):
    if userid is None:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    session = app.state.session_store.load(userid)
    if session is None:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    if session["phase"] != "bot_select":
        return RedirectResponse(url="/game", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "pages/select_bots.html",
        {
            "request": request,
            "bot_options": app.state.bot_options,
            "num_players": 1,
        },
    )


@app.post("/select-bots", response_class=RedirectResponse)
async def select_bots_submit(
    request: Request,
    userid: Optional[str] = Cookie(None),
):
    if userid is None:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    session = app.state.session_store.load(userid)
    if session is None:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    form = await request.form()
    bots = form.getlist("bots")
    app.state.session_store.set_bots(userid, bots)
    return RedirectResponse(url="/game", status_code=status.HTTP_302_FOUND)


@app.get("/game", response_class=HTMLResponse)
def enter_custom(
    request: Request,
    userid: Optional[str] = Cookie(None),
    username: Optional[str] = Cookie(None),
):
    if userid is None or username is None:
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        response.delete_cookie("userid")
        response.delete_cookie("username")
        return response

    session = app.state.session_store.load(userid)
    if session is None:
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        response.delete_cookie("userid")
        response.delete_cookie("username")
        return response

    if session["phase"] == "bot_select":
        return RedirectResponse(url="/select-bots", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "pages/index.html",
        {
            "request": request,
            "userid": userid,
            "username": username,
            "playerid": 0,
            "no_download": app.state.config["no_download"],
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    return Response(status_code=204)


async def process_data(data, websocket, ctx):
    if not ctx.ALT_STARTED:
        ctx.GLOB_THREAD = threading.Thread(
            target=per_user_game_loop, args=[ctx], daemon=True
        )
        ctx.GLOB_THREAD.start()

    try:
        pkg = json.loads(data)
    except Exception:
        return
    if "type" not in pkg:
        return

    logger.debug("ws recv: %s", pkg)
    if pkg["type"] == "player-join":
        player_join(ctx, websocket)
        await ctx.manager.send_dict({"type": "loading", "remain": 1}, websocket)
    elif pkg["type"] == "player-ready":
        ctx.strats[0].response = pkg
        ctx.strats[0].ready = True
    elif pkg["type"] == "player-move":
        ctx.strats[0].response = pkg
    elif pkg["type"] == "player-reset":
        logger.info("game reset requested by %s", ctx.userid)
        ctx.reset_game()
    else:
        logger.warning("unknown message type: %s", pkg["type"])


@app.websocket("/ws/{userid}")
async def websocket_endpoint(websocket: WebSocket, userid: str):
    logger.info("websocket connection from %s", userid)

    session = app.state.session_store.load(userid)
    if session is None:
        await websocket.accept()
        await websocket.send_json(json.dumps({"type": "invalid-session"}))
        await websocket.close(code=1000)
        return

    bots = json.loads(session["bots"]) if session["bots"] else []
    ctx = Context(
        userid=session["userid"],
        username=session["username"],
        bots=bots,
        history_folder=app.state.config["history_folder"],
    )
    app.state.active_games[userid] = ctx
    await ctx.manager.connect(websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            await process_data(raw, websocket, ctx)
    except WebSocketDisconnect:
        logger.info("client %s disconnected", ctx.username)
        ctx.disconnected = True
        if ctx.strats:
            ctx.strats[0].disconnected = True
        app.state.active_games.pop(userid, None)
