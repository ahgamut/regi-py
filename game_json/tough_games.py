import argparse
import datetime
import glob
import hashlib
import json
import os
import zipfile


def _digest_start_state(start):
    """Hash the opening deal: each player's cards + the three shared piles.

    Historically the pile lookups leaked the loop variable ``x0`` (the last
    player), so ``draw_pile``/``discard_pile``/``enemy_pile`` -- keys a player
    dict never has -- silently hashed as empty.  Fixed to read them from the
    game dict ``start``.
    """
    sub = []
    for player in start.get("players", []):
        sub.append(player.get("cards", []))
    sub.append(start.get("draw_pile", []))
    sub.append(start.get("discard_pile", []))
    sub.append(start.get("enemy_pile", []))

    raw = json.dumps(sub).encode("utf-8")
    h = hashlib.blake2b(digest_size=16)
    h.update(raw)
    return h.hexdigest()


def zip_members(z):
    # ``*.phases.json`` sidecars (run_benchmark --save-phases) are lists of phase
    # STRINGS, not event logs -- match game event logs only.
    return [
        name
        for name in z.namelist()
        if "game" in name
        and name.endswith(".json")
        and not name.endswith(".phases.json")
    ]


def get_metas(fname):
    bname = os.path.basename(fname)
    b0 = os.path.splitext(bname)[0]
    parts = b0.split("-")
    padding = ["game00", "team00", "sim00"]
    if len(parts) < 3:
        parts = parts + padding[len(parts) :]
    g, t, s = parts[:3]
    return bname, g, t, s.replace("sim", "s")


def game_digest(objs):
    start = None
    for o in objs:
        if o.get("event", "STATE") == "STARTGAME":
            start = o.get("game", dict())
            break
    if start is None:
        start = objs[0].get("game", dict())
    return _digest_start_state(start)


def proc_file(fname, hashes, z):
    try:
        fileobj = z.open(fname, "r")
        records = json.load(fileobj)
        hs = game_digest(records)
        # print(fname, hs, hs in hashes)

        for evt in records:
            if evt.get("event", "") == "POSTGAME":
                # print(evt, "progress" in evt)
                progress = evt["game"].get("progress", 0)
                return hs, fname, progress
        return hs, fname, 0
    except Exception as e:
        print("skipped", fname, e)
        return "", fname, 0


def check_tough_games(files, hashes, z, num_players, starts, out_fname):
    game_progs = dict()
    for fname in files:
        bname, game0, team, sim = get_metas(fname)
        game = int(game0.replace("game", ""))
        hs, _, progress = proc_file(fname, hashes, z)

        game_progs[game] = max(game_progs.get(game, 0), progress)
        if hs in hashes:
            print(game, team, sim, hs, "in hashes", progress)
            hashes.remove(hs)

    diff_games = list(sorted((z for z in game_progs.items()), key=lambda x: x[1]))

    easiest_5 = diff_games[-5:]
    toughest_5 = diff_games[:5]
    loc_mid = len(diff_games) // 2
    middle_5 = diff_games[loc_mid - 2 : loc_mid + 3]

    tiers = []

    print("Easiest", easiest_5)
    tiers.append({"name": "Easy", "phases": [starts[z[0]] for z in easiest_5]})
    print("Middle", middle_5)
    tiers.append({"name": "Medium", "phases": [starts[z[0]] for z in middle_5]})
    print("Toughest", toughest_5)
    tiers.append({"name": "Hard", "phases": [starts[z[0]] for z in toughest_5]})

    result = {"num_players": num_players, "tiers": tiers}
    with open(out_fname, "w") as of:
        json.dump(result, of, indent=4)


def main():
    parser = argparse.ArgumentParser("tough-games")
    parser.add_argument(
        "-s",
        "--source-hashes",
        default=None,
        help="file containing lines of game hashes to look at",
    )
    parser.add_argument(
        "-i",
        "--input-object",
        required=True,
        help="a ZIP archive of logs",
    )
    parser.add_argument(
        "-j",
        "--starts-json",
        required=True,
        help="a JSON created by make_phases.py",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="output JSON to store easiest/middle/toughest games",
    )
    d = parser.parse_args()

    z = zipfile.ZipFile(d.input_object, "r")
    files = zip_members(z)
    st_info = json.load(open(d.starts_json))
    starts = st_info["phases"]
    num_players = int(st_info["num_players"])

    if d.source_hashes is not None:
        hashes = set(x.strip() for x in open(d.source_hashes).readlines())
    else:
        hashes = set()

    check_tough_games(files, hashes, z, num_players, starts, d.output_file)


if __name__ == "__main__":
    main()
