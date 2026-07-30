import contextlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Optional
from uuid import uuid4

from fastapi import Cookie, Form
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi import Request
from fastapi import status
from fastapi.responses import Response
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse

###
from regi_py import get_strategy_map, GameState
from webdriver.common import (
    send_encoded,
    WebPlayerStrategy,
    WebPlayerLog,
    make_app,
    GameInterruptedError,
)

logger = logging.getLogger("regi")


###
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
    """Single-socket manager for one user's session."""

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
            await send_encoded(ws, message)

    async def broadcast_dict(self, message: dict):
        await self.send_dict(message)


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


app, templates = make_app(lifespan=lifespan)


###


class Context:
    def __init__(self, userid, username, bots, history_folder=None):
        self.manager = ConnectionManager()
        self.playerlog = WebPlayerLog(
            self.manager, self.resolve_name, history_folder=history_folder
        )
        self.game = GameState(self.playerlog)
        self.strats = []
        self.bots = bots
        self.userid = userid
        self.username = username
        self.ALT_STARTED = False
        self.GLOB_THREAD = None
        self.disconnected = False

    def resolve_name(self, player_id):
        # single-user session: the human is always player 0
        return self.username if player_id == 0 else None

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
    return templates.TemplateResponse(
        "pages/login.html", {"request": request, "require_password": True}
    )


@app.post("/login", response_class=RedirectResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if len(username) > 16 or len(password) > 16:
        return templates.TemplateResponse(
            "pages/login.html",
            {
                "request": request,
                "require_password": True,
                "error": "Username and password must be 16 characters or less.",
            },
        )
    if password != app.state.config["password"]:
        return templates.TemplateResponse(
            "pages/login.html",
            {"request": request, "require_password": True, "error": "Incorrect password."},
        )
    userid = str(uuid4())
    app.state.session_store.create(userid, username)
    response = RedirectResponse(url="/select-bots", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="userid", value=userid)
    response.set_cookie(key="username", value=username)
    return response


@app.get("/select-bots", response_class=HTMLResponse)
def select_bots_page(request: Request, userid: Optional[str] = Cookie(None)):
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
async def select_bots_submit(request: Request, userid: Optional[str] = Cookie(None)):
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

    logger.debug("ws recv: %s", pkg.get("type"))
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
        await send_encoded(websocket, {"type": "invalid-session"})
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
