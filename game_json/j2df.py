import json
import csv
import glob
import argparse
import os
import zipfile
import hashlib

IGNORE_EVENTS = ("STATE", "REPLENISH", "DRAWONE")

COLNAMES = [
    "alive_player_0",
    "alive_player_1",
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
    "n_cards",
    "num_cards_player_0",
    "num_cards_player_1",
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
    "game.players.0.alive",
    "game.players.0.cards",
    "game.players.0.id",
    "game.players.0.num_cards",
    "game.players.0.strategy",
    "game.players.1.alive",
    "game.players.1.cards",
    "game.players.1.id",
    "game.players.1.num_cards",
    "game.players.1.strategy",
    "game.players.2.alive",
    "game.players.2.cards",
    "game.players.2.id",
    "game.players.2.num_cards",
    "game.players.2.strategy",
    "game.players.3.alive",
    "game.players.3.cards",
    "game.players.3.id",
    "game.players.3.num_cards",
    "game.players.3.strategy",
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


def l1_list(lst):
    return "|".join([str(x) for x in lst])


def l2_list(lst):
    l0 = [l1_list(x) for x in lst]
    return ";".join(l0)


def proc_colname(obj, name):
    if "_player_0" in name:
        if "game" not in obj:
            return None
        key = name.replace("_player_0", "")
        return proc_colname(obj["game"]["players"][0], key)
    elif "_player_1" in name:
        if "game" not in obj:
            return None
        key = name.replace("_player_1", "")
        return proc_colname(obj["game"]["players"][1], key)
    elif "used_combos." in name:
        if "used_combos" not in obj:
            return None
        key = name.split(".")[-1]
        ll = [x[key] for x in obj["used_combos"]]
        return l2_list(ll)
    elif "combo." in name:
        if "combo" not in obj:
            return None
        n1 = name.split(".")[-1]
        ll = [x[n1] for x in obj["combo"]]
        return l1_list(ll)
    elif "players." in name:
        if "players" not in obj:
            return None
        _, pid, key = name.split(".")
        pid = intify(s)
        return proc_colname(obj["players"][pid], key)

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
    raw = json.dumps(objs).encode("utf-8")
    h = hashlib.blake2b(digest_size=32)
    h.update(raw)
    return h.hexdigest()


def group_games(objs, fname):
    count = 0
    results = [{"events": []}]
    for o in objs:
        results[count]["events"].append(o)
        if o.get("event", "STATE") == "POSTGAME":
            results.append({"events": []})
            count += 1

    for g in results[:count]:
        g["team"] = team_fixed(g["events"], fname)
        g["game"] = game_digest(g["events"])

    return results[:count]


def proc_file(fname, z=None):
    print(f"processing {fname}")
    bname, sim = get_metas(fname)
    if z is not None:
        try:
            logs = json.load(z.open(fname))
        except Exception as e:
            print("skipped", fname, e)
            return []
    else:
        logs = json.load(open(fname))

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
    parser = argparse.ArgumentParser("rowify-jsons")
    parser.add_argument(
        "-i",
        "--input-object",
        required=True,
        help="input a folder, JSON file, or ZIP file containing JSONS",
    )
    parser.add_argument("-o", "--output-csv", required=True, help="output csv")
    d = parser.parse_args()
    #
    if d.input_object.endswith(".zip"):
        z = zipfile.ZipFile(d.input_object, "r")
        files = [name for name in z.namelist() if "game" in name]
    elif d.input_object.endswith(".json"):
        z = None
        files = [d.input_object]
    else:
        z = None
        files = glob.glob(os.path.join(d.input_object, "game*.json"))
    #
    with open(d.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FILEMETA + COLNAMES + PLAYERINFO)
        for file in files:
            rows = proc_file(file, z)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
