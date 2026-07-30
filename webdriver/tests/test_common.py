"""Opt-in tests for the shared webapp helpers in ``webdriver/common.py``.

Deliberately OUTSIDE the main pytest ``testpaths=["tests"]`` so they never run in
the default suite. Run explicitly:  ``pytest webdriver/tests``. They need the
webapp deps (fastapi/jinja2) + the built ``regi_py`` extension, but NOT torch.
"""
import asyncio
import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("fastapi", reason="webapp deps (fastapi) not installed")
pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

from webdriver import common  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# --- recommender spec parsing --------------------------------------------- #
def test_parse_reco_spec_valid():
    assert common.parse_reco_spec("brute-128") == ("brute", 128)
    assert common.parse_reco_spec("mcts-64") == ("mcts", 64)
    assert common.parse_reco_spec("adzmulti-0") == ("adzmulti", 0)


@pytest.mark.parametrize("bad", ["brute", "brute-x", "mcts-", "-5", "adzmulti"])
def test_parse_reco_spec_invalid(bad):
    with pytest.raises(ValueError):
        common.parse_reco_spec(bad)


def test_validate_reco_nn_requires_weights(tmp_path):
    assert common.validate_reco("brute-128", None) == ("brute", 128)
    assert common.validate_reco("mcts-64", None) == ("mcts", 64)
    with pytest.raises(ValueError):  # NN net, no weights
        common.validate_reco("adzmulti-0", None)
    with pytest.raises(ValueError):  # NN net, missing weights file
        common.validate_reco("adzmulti-0", str(tmp_path / "nope.pt"))
    wt = tmp_path / "w.pt"
    wt.write_bytes(b"x")
    assert common.validate_reco("adzmulti-0", str(wt)) == ("adzmulti", 0)


def test_make_recommender_builtins():
    from regi_py.strats import BruteSamplingStrategy
    from regi_py.strats.mcts_explorer import MCTSExplorerStrategy

    assert isinstance(common.make_recommender("brute", 4, None), BruteSamplingStrategy)
    assert isinstance(common.make_recommender("mcts", 4, None), MCTSExplorerStrategy)


def test_builtin_recommender_imports_no_torch():
    # in a fresh interpreter, building a brute recommender must not import torch
    code = (
        "import sys, webdriver.common as c;"
        "r = c.make_recommender('brute', 4, None);"
        "assert type(r).__name__ == 'BruteSamplingStrategy';"
        "assert 'torch' not in sys.modules, 'torch was imported for a builtin reco'"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr


# --- username enrichment --------------------------------------------------- #
def test_enrich_with_usernames_uses_resolver():
    resolve = lambda pid: {0: "alice", 1: "bob"}.get(pid)
    data = {
        "player": {"id": 0},
        "players": [{"id": 1}, {"id": 2}],
        "nested": {"active_player": {"id": 1}},
    }
    common.enrich_with_usernames(data, resolve)
    assert data["player"]["username"] == "alice"
    assert data["players"][0]["username"] == "bob"
    assert "username" not in data["players"][1]  # id 2 unresolved
    assert data["nested"]["active_player"]["username"] == "bob"


# --- lighter wire payload -------------------------------------------------- #
def test_trim_for_wire_drops_game_on_board_events():
    atk = {"event": "ATTACK", "combo": [], "game": {"players": []}}
    wire = common.trim_for_wire(atk)
    assert "game" not in wire          # dropped on the wire
    assert "game" in atk               # source object untouched (save/download)


@pytest.mark.parametrize("event", ["STATE", "STARTGAME", "ENDGAME", "POSTGAME", "REDIRECT"])
def test_trim_for_wire_keeps_game_on_state_events(event):
    obj = {"event": event, "game": {"x": 1}}
    assert common.trim_for_wire(obj) is obj  # unchanged, same object


def test_trim_for_wire_no_game_passthrough():
    obj = {"event": "DRAWONE", "player": {"id": 1}}
    assert common.trim_for_wire(obj) is obj


# --- wire encoding --------------------------------------------------------- #
def test_send_encoded_single_encode():
    class FakeWS:
        def __init__(self):
            self.sent = []

        async def send_text(self, s):
            self.sent.append(s)

    ws = FakeWS()
    msg = {"type": "log", "data": {"a": 1, "b": [1, 2]}}
    asyncio.run(common.send_encoded(ws, msg))
    assert len(ws.sent) == 1
    raw = ws.sent[0]
    assert isinstance(raw, str)
    once = json.loads(raw)          # a SINGLE decode yields the dict...
    assert once == msg
    assert not isinstance(once, str)  # ...not another JSON string (no double-encode)
