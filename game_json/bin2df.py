import json
import csv
import glob
import argparse
import os
import zipfile
import hashlib
import msgpack

#
from regi_py.strats.mcts_explorer import MCTSNodeInfo
from regi_py.logging.utils import dump_debug, dump_card
from regi_py.core import *
from regi_py.logging import DummyLog

IGNORE_EVENTS = ("STATE", "REPLENISH", "DRAWONE")

COLNAMES = [
    "combo.strength",
    "combo.value",
    "damage",
    "enemy.hp",
    "enemy.strength",
    "enemy.value",
    "event",
    "fullblock",
    "game.active_player.alive",
    "game.active_player.cards",
    "game.active_player.id",
    "game.active_player.num_cards",
    "game.active_player.strategy",
    "game.active_player_id",
    "game.current_block",
    "game.current_enemy.hp",
    "game.current_enemy.strength",
    "game.current_enemy.value",
    "game.discard_pile",
    "game.discard_pile_size",
    "game.draw_pile",
    "game.draw_pile_size",
    "game.enemy_pile",
    "game.enemy_pile_size",
    "game.hand_size",
    "game.num_players",
    "game.past_yields",
    "game.phase_attacking",
    "game.phase_count",
    "game.progress",
    "game.status",
    "maxblock",
    "player.alive",
    "player.cards",
    "player.id",
    "player.num_cards",
    "player.strategy",
    "strategy",
    "used_combos.strength",
    "used_combos.value",
]

PLAYERINFO = [
    "game.players.0.id",
    "game.players.0.alive",
    "game.players.0.strategy",
    "game.players.0.num_cards",
    "game.players.0.cards",
    "game.players.1.id",
    "game.players.1.alive",
    "game.players.1.strategy",
    "game.players.1.num_cards",
    "game.players.1.cards",
    "game.players.2.id",
    "game.players.2.alive",
    "game.players.2.strategy",
    "game.players.2.num_cards",
    "game.players.2.cards",
    "game.players.3.id",
    "game.players.3.alive",
    "game.players.3.strategy",
    "game.players.3.num_cards",
    "game.players.3.cards",
]

FILEMETA = [
    "file",
    "game",
    "team",
    "sim",
]


def intify(s):
    try:
        return int(s)
    except Exception as e:
        print("unable to int", s, e)
        return s


def l1_list(lst, ch="|"):
    return ch.join([str(x) for x in lst])


def l2_list(lst):
    l0 = [l1_list(x, "&") for x in lst]
    return ";".join(l0)


def proc_colname(obj, name):
    if "used_combos." in name:
        if "used_combos" not in obj.get("game", ""):
            return None
        key = name.split(".")[-1]
        ll = []
        if key == "value":
            for x0 in obj["game"]["used_combos"]:
                ll.append([x1[key] for x1 in x0])
        else:
            for x0 in obj["game"]["used_combos"]:
                ll.append([x1[key] for x1 in x0])
        return l2_list(ll)
    elif "combo." in name:
        if "combo" not in obj:
            return None
        n1 = name.split(".")[-1]
        ll = [x[n1] for x in obj["combo"]]
        return l1_list(ll)
    elif "players." in name:
        if "game" not in obj:
            return None
        if "players" not in obj["game"]:
            return None
        _, _, pid, key = name.split(".")
        pid = intify(pid)
        if pid >= len(obj["game"]["players"]):
            return None
        return proc_colname(obj["game"]["players"][pid], key)

    subs = name.split(".")
    o0 = obj
    assert isinstance(o0, dict)
    for s in subs:
        # print(subs, s, o0)
        o0 = o0.get(s)
        if o0 is None:
            break
    #
    if o0 is None:
        return o0
    if isinstance(o0, bool):
        o0 = "TRUE" if o0 else "FALSE"
    if isinstance(o0, list):
        o0 = l1_list(o0)
    return o0


def proc_event(event, file, game, team, sim):
    row = [file, game, team, sim]
    for colname in COLNAMES + PLAYERINFO:
        row.append(proc_colname(event, colname))
    return row


def get_metas(fname):
    bname = os.path.basename(fname)
    b0 = os.path.splitext(bname)[0]
    parts = b0.split("-")
    padding = ["game00", "team00", "sim00"]
    if len(parts) < 3:
        parts = parts + padding[len(parts) :]
    g, t, s = parts[:3]
    return bname, s.replace("sim", "s")


def team_fixed(objs, fname):
    teams = set()
    for o in objs:
        if "game" in o:
            if "players" in o["game"]:
                t0 = [x["strategy"] for x in o["game"]["players"]]
                t1 = "|".join(t0)
                teams.add(t1)

    teams = list(teams)
    assert (
        len(teams) == 1
    ), f"events in {fname} don't have a fixed team {teams}. multiple games?"
    return teams[0]


def game_digest(objs):
    start = objs[0].get("game", dict())
    sub = []
    for x0 in start.get("players", []):
        sub.append(x0.get("cards", []))
    sub.append(x0.get("draw_pile", []))
    sub.append(x0.get("discard_pile", []))
    sub.append(x0.get("enemy_pile", []))

    raw = json.dumps(sub).encode("utf-8")
    h = hashlib.blake2b(digest_size=16)
    h.update(raw)
    # postprocess game data
    for i, o in enumerate(objs):
        o["game"]["phase_count"] = i
        o["game"]["active_player"]["strategy"] = "mcts-explorer"
        for p in o["game"]["players"]:
            p["strategy"] = "mcts-explorer"
    return h.hexdigest()

def argmax(lst):
    ind = 0
    mvx = -100
    for i, x in enumerate(lst):
        if x > mvx:
            ind = i
            mvx = x
    return ind

def multi(game, cmb):
    if len(game.enemy_pile) == 0:
        return 1
    #
    enemy = game.enemy_pile[0]
    #
    cmb_has_clubs = False
    for card in cmb:
        if "\u2993" in card["value"]:
            cmb_has_clubs = True
            break
    if not cmb_has_clubs:
        return 1
    #
    joker_nerf = False
    for combo in game.used_combos:
        for card in combo:
            if card.suit == Suit.GLITCH:
                joker_nerf = True
    if joker_nerf or enemy.suit != Suit.CLUBS:
        return 2
    return 1


def phase_str_to_game_dct(info):
    phase = PhaseInfo.from_string(info["phase"])
    if phase.phase_attacking:
        event = "ATTACK"
    else:
        event = "DEFEND"
    log = DummyLog()
    game = GameState(log)
    strat = RandomStrategy()
    for _ in range(phase.num_players):
        game.add_player(strat)
    game._init_phaseinfo(phase)
    data = dump_debug(game)
    acp = game.players[game.active_player]

    r = dict()
    r["event"] = event
    r["game"] = data
    r["enemy"] = data["current_enemy"]
    r["player"] = data["active_player"]
    r["strategy"] = "mcts-explorer"
    r["used_combos"] = data["used_combos"]

    dmg = 0
    if len(info["combos"]) > 0:
        selected_ind = int(argmax(info["N1"]))
        selected_combo = info["combos"][selected_ind]
        # print(info["combos"], info["N1"], selected_ind, selected_combo)
        cmb = []
        for x in acp.cards:
            if str(x) in selected_combo:
                cmb.append(dump_card(x))
        r["combo"] = cmb
        dmg = sum(x["strength"] for x in cmb) * multi(game, cmb)
        if event == "ATTACK":
            data["used_combos"].append(cmb)
    if event == "ATTACK":
        r["damage"] = dmg
    else:
        r["maxblock"] = sum(x.strength for x in acp.cards)
    r["last"] = phase.game_endvalue != 0
    return r


def group_games(objs, fname):
    count = 0
    results = [{"events": []}]
    for o in objs:
        try:
            r = phase_str_to_game_dct(o)
            results[count]["events"].append(r)
            if r["last"]:
                results.append({"events": []})
                count += 1
        except Exception as e:
            print("state loading error:", e)

    for g in results[:count]:
        g["game"] = game_digest(g["events"])
        g["team"] = team_fixed(g["events"], fname)

    return results[:count]


def proc_file(fname, z=None):
    print(f"processing {fname}")
    bname, sim = get_metas(fname)
    try:
        if z is not None:
            logs = msgpack.Unpacker(z.open(fname, "rb"))
        else:
            logs = msgpack.Unpacker(open(fname, "rb"))
    except Exception as e:
        print("skipped", fname, e)
        return []

    games = group_games(logs, bname)
    rows = []
    for g in games:
        team = g["team"]
        game = g["game"]
        for e in g["events"]:
            if e.get("event", "STATE") in IGNORE_EVENTS:
                continue
            rows.append(proc_event(e, bname, game, team, sim))
    return rows


def main():
    parser = argparse.ArgumentParser("rowify-bins")
    parser.add_argument(
        "-i",
        "--input-object",
        required=True,
        help="input a folder, BIN file, or ZIP file containing NodeInfo",
    )
    parser.add_argument("-o", "--output-csv", required=True, help="output csv")
    d = parser.parse_args()
    #
    if d.input_object.endswith(".zip"):
        z = zipfile.ZipFile(d.input_object, "r")
        files = [name for name in z.namelist() if name.endswith(".bin")]
    elif d.input_object.endswith(".bin"):
        z = None
        files = [d.input_object]
    else:
        z = None
        files = glob.glob(os.path.join(d.input_object, "*.bin"))
    #
    with open(d.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FILEMETA + COLNAMES + PLAYERINFO)
        for file in files:
            rows = proc_file(file, z)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
