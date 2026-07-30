import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import Optional
from uuid import uuid4

#
import uvicorn
from fastapi import Cookie, Form
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
from common import (
    send_encoded,
    WebPlayerStrategy,
    WebPlayerLog,
    make_app,
    make_recommender,
    validate_reco,
)

logger = logging.getLogger("regi")


###
class ConnectionManager:
    """Multiplayer manager: a list of connected sockets with broadcast."""

    def __init__(self, num_players, num_bots):
        self.active_connections: list[WebSocket] = []
        self.num_players = num_players
        self.num_bots = num_bots

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        await websocket.close(code=1000, reason=None)
        self.active_connections.remove(websocket)

    async def send_dict(self, message: dict, websocket: WebSocket):
        await send_encoded(websocket, message)

    async def send_string(self, message: str, websocket: WebSocket):
        await self.send_dict({"type": "message", "data": message}, websocket)

    async def broadcast_dict(self, message: dict):
        for connection in self.active_connections:
            await self.send_dict(message, connection)

    async def broadcast_string(self, message: str):
        await self.broadcast_dict({"type": "message", "data": message})


###
app, templates = make_app()


class Context:
    def __init__(
        self,
        num_players,
        bots,
        password,
        skip_bots=False,
        no_download=False,
        history_folder=None,
        recommender=None,
    ):
        self.manager = ConnectionManager(num_players, len(bots))
        self.playerlog = WebPlayerLog(
            self.manager, self.resolve_name, history_folder=history_folder
        )
        self.game = GameState(self.playerlog)
        self.num_players = num_players
        self.strats = []
        self.bots = bots
        self.password = password
        self.skip_bots = skip_bots
        self.no_download = no_download
        self.recommender = recommender
        self.userids = []
        self.usernames = {}
        self.ALT_STARTED = False
        self.GLOB_THREAD = None
        self.bot_options = list(get_strategy_map(rl_mods=False).keys())

    def resolve_name(self, player_id):
        if 0 <= player_id < len(self.userids):
            return self.usernames.get(self.userids[player_id])
        return None

    @property
    def needs_bot_selection(self):
        if len(self.bots) > 0:
            return False
        if self.skip_bots and self.num_players >= 2:
            return False
        return True

    def set_bots(self, bots):
        self.bots = bots
        self.manager.num_bots = len(bots)

    def load_game(self):
        assert app.state.CTX.ALT_STARTED
        strategy_map = get_strategy_map(rl_mods=False)
        if len(self.strats) != self.num_players + len(self.bots):
            for i in range(self.num_players):
                self.game.add_player(self.strats[i])
            for b in self.bots:
                strat = strategy_map[b]()
                self.strats.append(strat)
                self.game.add_player(self.strats[-1])
        else:
            for s in self.strats:
                self.game.add_player(s)

        print("starting with", [x.__strat_name__ for x in self.strats])
        assert len(self.strats) >= 2
        assert len(self.strats) <= 4
        if self.num_players == 1 and len(self.bots) > 0:
            print("solo player with bots, skipping ready check")
            self.strats[0].ready = True
        self.game.initialize()
        if not (self.num_players == 1 and len(self.bots) > 0):
            t = 0
            while t != self.num_players:
                time.sleep(1)
                t = sum(p.ready for p in self.strats[: self.num_players])
                print(t, "players ready")
        self.game.start_loop()

    def end_game(self):
        assert self.game is not None
        del self.game
        self.game = None

    def reset_game(self):
        self.game = GameState(self.playerlog)


def game_loop():
    CTX = app.state.CTX
    CTX.ALT_STARTED = True
    while True:
        while CTX.needs_bot_selection:
            print("waiting for bot selection")
            time.sleep(1)

        while CTX.num_players > len(CTX.strats):
            print("we are waiting for players")
            time.sleep(1)

        CTX.load_game()
        CTX.end_game()

        while CTX.game is None:
            time.sleep(1)


def player_join(userid, username, websocket):
    ctx = app.state.CTX
    if len(ctx.strats) == len(ctx.userids):
        return
    strat = WebPlayerStrategy(
        userid, username, websocket, ctx, recommender=ctx.recommender
    )
    ctx.strats.append(strat)


###


# pylint: disable=W0613
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(
        "pages/login.html",
        {"request": request, "require_password": app.state.CTX.password is not None},
    )


@app.post("/login", response_class=RedirectResponse)
async def login(request: Request, username: str = Form(...), password: str = Form("")):
    require_password = app.state.CTX.password is not None
    if len(username) > 16 or len(password) > 16:
        return templates.TemplateResponse(
            "pages/login.html",
            {
                "request": request,
                "require_password": require_password,
                "error": "Username and password must be 16 characters or less.",
            },
        )
    if require_password and password != app.state.CTX.password:
        return templates.TemplateResponse(
            "pages/login.html",
            {
                "request": request,
                "require_password": require_password,
                "error": "Incorrect password.",
            },
        )
    userid = str(uuid4())
    app.state.CTX.userids.append(userid)
    app.state.CTX.usernames[userid] = username
    if app.state.CTX.needs_bot_selection and len(app.state.CTX.userids) == 1:
        redirect_url = "/select-bots"
    elif app.state.CTX.needs_bot_selection:
        redirect_url = "/wait-bots"
    else:
        redirect_url = "/game"
    response = RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="userid", value=userid)
    response.set_cookie(key="username", value=username)
    return response


@app.get("/select-bots", response_class=HTMLResponse)
def select_bots_page(request: Request, userid: Optional[str] = Cookie(None)):
    if userid is None or userid not in app.state.CTX.userids:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "pages/select_bots.html",
        {
            "request": request,
            "bot_options": app.state.CTX.bot_options,
            "num_players": app.state.CTX.num_players,
        },
    )


@app.get("/wait-bots", response_class=HTMLResponse)
def wait_bots_page(request: Request, userid: Optional[str] = Cookie(None)):
    if userid is None or userid not in app.state.CTX.userids:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    if not app.state.CTX.needs_bot_selection:
        return RedirectResponse(url="/game", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("pages/wait_bots.html", {"request": request})


@app.get("/api/bots-ready", response_class=JSONResponse)
def bots_ready():
    return JSONResponse({"ready": not app.state.CTX.needs_bot_selection})


@app.post("/select-bots", response_class=RedirectResponse)
async def select_bots_submit(request: Request, userid: Optional[str] = Cookie(None)):
    if userid is None or userid not in app.state.CTX.userids:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    form = await request.form()
    bots = form.getlist("bots")
    app.state.CTX.set_bots(bots)
    return RedirectResponse(url="/game", status_code=status.HTTP_302_FOUND)


@app.get("/game", response_class=HTMLResponse)
def enter_custom(
    request: Request,
    userid: Optional[str] = Cookie(None),
    username: Optional[str] = Cookie(None),
):
    if userid is None or username is None or userid not in app.state.CTX.userids:
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        response.delete_cookie("userid")
        response.delete_cookie("username")
        return response

    return templates.TemplateResponse(
        "pages/index.html",
        {
            "request": request,
            "userid": userid,
            "username": username,
            "playerid": app.state.CTX.userids.index(userid),
            "no_download": app.state.CTX.no_download,
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    return Response(status_code=204)


async def process_data(data, websocket):
    CTX = app.state.CTX
    #
    if not CTX.ALT_STARTED:
        CTX.GLOB_THREAD = threading.Thread(group=None, target=game_loop, args=[])
        CTX.GLOB_THREAD.start()
    #
    try:
        pkg = json.loads(data)
    except Exception:
        return
    if "type" not in pkg:
        return

    logger.debug("ws recv: %s", pkg.get("type"))
    if pkg["type"] == "player-join":
        username = CTX.usernames.get(pkg["userid"], "Unknown Player")
        player_join(pkg["userid"], username, websocket)
        await CTX.manager.send_dict({"type": "loading", "remain": 1}, websocket)
    elif pkg["type"] == "player-ready":
        playerid = CTX.manager.active_connections.index(websocket)
        CTX.strats[playerid].response = pkg
        CTX.strats[playerid].ready = True
    elif pkg["type"] == "player-move":
        playerid = CTX.manager.active_connections.index(websocket)
        CTX.strats[playerid].response = pkg
    elif pkg["type"] == "player-reset":
        CTX.reset_game()
    else:
        logger.warning("unknown message type: %s", pkg.get("type"))


@app.websocket("/ws/{userid}")
async def websocket_endpoint(websocket: WebSocket, userid: str):
    if userid not in app.state.CTX.userids:
        await websocket.accept()
        await send_encoded(websocket, {"type": "invalid-session"})
        await websocket.close(code=1000)
        return
    await app.state.CTX.manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            await process_data(raw, websocket)
    except WebSocketDisconnect:
        await app.state.CTX.manager.disconnect(websocket)
        username = app.state.CTX.usernames.get(userid, "Unknown Player")
        await app.state.CTX.manager.broadcast_string(f"Client {username} left the chat")


def make_CTX(app, d):
    if app.state.CTX is not None:
        return
    # recommendations are opt-in: build the recommender ONCE (shared across players)
    # only when --reco was given; torch is imported only if d.reco names an NN net
    # (brute/mcts stay torch-free)
    recommender = None
    if d.reco is not None:
        recommender = make_recommender(d.reco_name, d.reco_iters, d.reco_weights)
    app.state.CTX = Context(
        d.num_players,
        d.bots,
        d.password,
        d.skip_bots,
        d.no_download,
        d.history_folder,
        recommender=recommender,
    )
    pw = "no password" if d.password is None else "password required"
    reco = "off" if d.reco is None else d.reco
    print(
        f"\n\n\nreco={reco} ({pw})\n"
        f"Go to http://{d.host}:{d.port} on your browser to view webserver\n\n\n",
        sep="",
    )


def load_args():
    strategy_map = get_strategy_map(rl_mods=False)
    parser = argparse.ArgumentParser(
        prog="regi-webserver",
        description="FastAPI websockets server for regi-py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="host for uvicorn server")
    parser.add_argument("--port", default=8888, help="port for uvicorn server")
    parser.add_argument(
        "-n", "--num-players", type=int, default=1, help="number of players"
    )
    parser.add_argument(
        "--password",
        default=None,
        help="password required to join the game (default: no password)",
    )
    parser.add_argument(
        "-b",
        "--add-bot",
        dest="bots",
        action="append",
        default=[],
        help="bot options: " + ",".join(strategy_map),
    )
    parser.add_argument(
        "--skip-bots",
        dest="skip_bots",
        action="store_true",
        default=False,
        help="skip bot selection when 2-4 human players are specified",
    )
    parser.add_argument(
        "--no-download",
        dest="no_download",
        action="store_true",
        default=False,
        help="hide the Download JSON button at end of game",
    )
    parser.add_argument(
        "--history-folder",
        dest="history_folder",
        default=None,
        help="folder to save game history JSON files after each game",
    )
    parser.add_argument(
        "--reco",
        dest="reco",
        default=None,
        help="enable move recommendations, as NAME-ITERS (e.g. brute-128, mcts-64, "
        "basic-0, adzmulti-64). Off by default. NAME is 'brute'/'mcts' or a trained "
        "net name; ITERS is the search budget (0 => search-free direct-net for NN "
        "nets). NN nets require --reco-weights and are the only case that imports torch.",
    )
    parser.add_argument(
        "--reco-weights",
        dest="reco_weights",
        default=None,
        help="path to a .pt checkpoint (required when --reco names an NN net)",
    )
    d = parser.parse_args()
    total_players = d.num_players + len(d.bots)

    if d.skip_bots and d.num_players < 2:
        print("ERROR --skip-bots requires at least 2 human players (-n 2 or more)\n\n")
        parser.print_help()
        sys.exit(1)
    if total_players > 4:
        print("ERROR can't have more than 4 players!\n\n")
        parser.print_help()
        sys.exit(1)
    # recommendations are opt-in: only validate/parse a spec when --reco is given
    d.reco_name, d.reco_iters = None, None
    if d.reco is not None:
        try:
            d.reco_name, d.reco_iters = validate_reco(d.reco, d.reco_weights)
        except ValueError as err:
            print(f"ERROR {err}\n\n")
            parser.print_help()
            sys.exit(1)
    return d


def main() -> None:
    app.state.CTX = None
    d = load_args()
    make_CTX(app, d)

    uvicorn.run(
        "__main__:app",
        host=d.host,
        port=int(d.port),
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
