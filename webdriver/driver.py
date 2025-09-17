import argparse
import datetime
import json
import logging
import os
import sys
import threading
import time
import tempfile
import traceback
from logging import FileHandler
from typing import Optional
from uuid import uuid4

#
import anyio
import uvicorn
import fastapi

#
from anyio.from_thread import BlockingPortalProvider
from anyio import from_thread, to_thread
from fastapi import Cookie
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi import Request
from fastapi import status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

###
from regi_py.strats import BaseStrategy, DamageStrategy
from regi_py import RegiEncoder, JSONBaseLog, GameState, GameStatus


###
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        await websocket.close(code=1000, reason=None)
        self.active_connections.remove(websocket)

    async def send_string(self, message: str, websocket: WebSocket):
        result = {"type": "message", "data": message}
        await self.send_dict(result, websocket)

    async def send_dict(self, message: dict, websocket: WebSocket):
        raw = json.dumps(message, cls=RegiEncoder)
        await websocket.send_json(raw)

    async def broadcast_string(self, message: str):
        result = {"type": "message", "data": message}
        for connection in self.active_connections:
            await self.send_dict(result, connection)

    async def broadcast_dict(self, message: dict):
        for connection in self.active_connections:
            await self.send_dict(message, connection)


class WebPlayerStrategy(BaseStrategy):
    __strat_name__ = "player-webui"

    def __init__(self, userid, websocket):
        super().__init__()
        self.userid = userid
        self.websocket = websocket
        self.portal_provider = BlockingPortalProvider()
        self.response = None
        self.ready = False

    @staticmethod
    async def comms_twoway(self, websocket, obj):
        print("sending", obj)
        await CTX.manager.send_dict(obj, websocket)

        while self.response is None:
            print(self.userid, "waiting", obj["type"])
            await anyio.sleep(1)

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
        print("player: ", player.id, "available attacks: ", combos)
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
        print("player: ", player.id, "available defenses: ", combos)
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


class WebPlayerLog(JSONBaseLog):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.count = 0
        self.portal_provider = BlockingPortalProvider()

    @staticmethod
    async def log_actual(manager, obj):
        result = {}
        result["type"] = "log"
        result["data"] = obj
        await manager.broadcast_dict(result)

    def log(self, obj):
        with self.portal_provider as portal:
            portal.call(WebPlayerLog.log_actual, self.manager, obj)
        self.count += 1


###
app = FastAPI(docs_url=None)
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

# if os.getenv("TMPDIR") is not None:
#    DEBUG_DIR_NAME = os.getenv("TMPDIR")
# else:
# DEBUG_DIR = tempfile.TemporaryDirectory(prefix="regipy-")  # pylint: disable=R1732
# DEBUG_DIR_NAME = DEBUG_DIR.name


def debugwrap(*args):
    if len(args) > 1:
        folder = args[:-1]
        os.makedirs(os.path.join(DEBUG_DIR_NAME, *folder), exist_ok=True)
    return os.path.join(DEBUG_DIR_NAME, *args)


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
    def __init__(self):
        self.manager = ConnectionManager()
        self.playerlog = WebPlayerLog(self.manager)
        self.game = GameState(self.playerlog)
        self.n_players = 1
        self.strats = []
        self.userids = []
        self.ALT_STARTED = False
        self.FUG_RESPONSE = None
        self.GLOB_THREAD = None

    def load_game(self):
        # assert len(self.userids) == self.n_players
        assert CTX.ALT_STARTED
        for i in range(self.n_players):
            self.game.add_player(self.strats[i])

        self.strats.append(DamageStrategy())
        self.game.add_player(self.strats[-1])

        self.game.initialize()
        self.game.start_loop()

    def end_game(self):
        assert self.game is not None
        del self.game
        self.game = None
        self.strats.clear()

    def reset_game(self):
        self.game = GameState(self.playerlog)


CTX = Context()


def game_loop():
    CTX.ALT_STARTED = True
    while True:
        while CTX.n_players > len(CTX.strats):
            # print("hello")
            # print(CTX.n_players, CTX.strats)
            time.sleep(1)

        print("OMG! Game loop can start???")
        CTX.load_game()
        CTX.end_game()

        # game should have ended
        while CTX.game is None:
            time.sleep(1)


def player_join(userid, websocket):
    if len(CTX.strats) == len(CTX.userids):
        return
    strat = WebPlayerStrategy(
        userid,
        websocket,
    )
    CTX.strats.append(strat)


###


# pylint: disable=W0613
@app.get("/", response_class=HTMLResponse)
def enter_custom(request: Request):
    client = request.client
    userid = str(uuid4())
    CTX.userids.append(userid)
    response = templates.TemplateResponse(
        "pages/index.html",
        {"request": request, "userid": userid, "playerid": CTX.userids.index(userid)},
    )
    return response


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    return Response(status_code=204)


async def process_data(data, websocket):
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

    print(pkg)
    if pkg["type"] == "player-join":
        player_join(pkg["userid"], websocket)
        await CTX.manager.send_dict({"type": "loading", "remain": 1}, websocket)
    elif pkg["type"] in ["player-ready"]:
        playerid = CTX.manager.active_connections.index(websocket)
        CTX.strats[playerid].response = pkg
        CTX.strats[playerid].ready = True
    elif pkg["type"] in ["player-move"]:
        playerid = CTX.manager.active_connections.index(websocket)
        CTX.strats[playerid].response = pkg
    elif pkg["type"] in ["player-reset"]:
        print("should reset")
        CTX.reset_game()
    else:
        print("FUG")


@app.websocket("/ws/{userid}")
async def websocket_endpoint(websocket: WebSocket, userid: str):
    print("got message from ", userid)
    await CTX.manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            await process_data(raw, websocket)
    except WebSocketDisconnect:
        # FUG
        await CTX.manager.disconnect(websocket)
        await CTX.manager.broadcast_string(f"Client {userid} left the chat")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="regi-webserver",
        description="FastAPI websockets server for regi-py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="host for uvicorn server")
    parser.add_argument("--port", default=8888, help="port for uvicorn server")
    parser.add_argument(
        "-n", "--num-players", type=int, default=2, help="number of players"
    )
    d = parser.parse_args()
    print(
        f"\n\n\nTemporary files ignored\n",
        f"Go to http://{d.host}:{d.port} on your browser to view webserver\n\n\n",
        sep="",
    )
    #
    uvicorn.run(
        "driver:app",
        host=d.host,
        port=int(d.port),
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
