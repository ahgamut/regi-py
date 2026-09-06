"""Turn Regicide game logs into flat CSV rows -- one schema, two sources.

This merges the former ``j2df.py`` (JSON event logs) and ``bin2df.py`` (msgpack
MCTS ``NodeInfo`` records), which were ~90% identical.  The dict schema and every
row-building helper live here exactly once; the parts that genuinely differ by
source (how a file is loaded, how records become game events, and how a game's
digest is computed) are isolated on a small ``LogSource`` type.

The column schema (``COLNAMES``/``PLAYERINFO``) is *derived* from the single
serialization source-of-truth in :mod:`regi_py.serialize` rather than hardcoded,
so adding a field to a game/player/enemy/combo dict shows up here automatically.

Output can go to a CSV (``-o``) and/or a SQLite database (``--sqlite``); both
share the one row schema, so each file is parsed once and its rows fanned out to
every requested sink.  ``sqlite3`` is in the stdlib, so the DB path needs no extra
dependency.

``j2df.py`` and ``bin2df.py`` remain as thin CLI shims that call :func:`main`
with the matching source, so existing command lines keep working.  ``msgpack`` is
imported lazily inside the msgpack source, so the JSON path needs no msgpack.
"""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import sqlite3
import zipfile

from regi_py.serialize import (
    CARD_FIELDS,
    ENEMY_FIELDS,
    PLAYER_FIELDS,
    GAME_FIELDS,
    DEBUG_EXTRA_FIELDS,
)

IGNORE_EVENTS = ("STATE", "REPLENISH", "DRAWONE")
MAX_PLAYERS = 4

# --------------------------------------------------------------------------- #
# schema -- derived from the serialize tuples (regi_py.serialize.*_FIELDS)
# --------------------------------------------------------------------------- #
# event-record scalars that are not part of any serialized game object
EVENT_SCALARS = ("damage", "event", "fullblock", "maxblock", "strategy")
# game object fields that are expanded into their own sub-columns rather than a
# flat ``game.<field>`` column
_GAME_EXPANDED = ("active_player", "players", "used_combos", "current_enemy")
GAME_SCALAR_FIELDS = (
    tuple(f for f in GAME_FIELDS if f not in _GAME_EXPANDED) + DEBUG_EXTRA_FIELDS
)

# the flat event columns, sorted (the historical, R-facing order)
COLNAMES = sorted(
    list(EVENT_SCALARS)
    + [f"combo.{f}" for f in CARD_FIELDS]
    + [f"used_combos.{f}" for f in CARD_FIELDS]
    + [f"enemy.{f}" for f in ENEMY_FIELDS]
    + [f"game.current_enemy.{f}" for f in ENEMY_FIELDS]
    + [f"player.{f}" for f in PLAYER_FIELDS]
    + [f"game.active_player.{f}" for f in PLAYER_FIELDS]
    + [f"game.{f}" for f in GAME_SCALAR_FIELDS]
)

# per-player columns keep their historical presentation order (a permutation of
# PLAYER_FIELDS), grouped by player id
PLAYERINFO_FIELD_ORDER = ("id", "alive", "strategy", "num_cards", "cards")
PLAYERINFO = [
    f"game.players.{pid}.{field}"
    for pid in range(MAX_PLAYERS)
    for field in PLAYERINFO_FIELD_ORDER
]

# ``run_id`` tags every row with the --run-id (a constant per invocation, e.g. an r1
# vs r2 benchmark run); it leads the file-level metadata columns.
FILEMETA = ["run_id", "file", "game", "team", "sim"]


# --------------------------------------------------------------------------- #
# shared row-building helpers (identical across both former parsers)
# --------------------------------------------------------------------------- #
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


def proc_event(event, run_id, file, game, team, sim):
    row = [run_id, file, game, team, sim]
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


# --------------------------------------------------------------------------- #
# sources: the parts that genuinely differ between JSON logs and msgpack records
# --------------------------------------------------------------------------- #
class LogSource:
    """A log format: how to find/read files, split games, and digest them."""

    name = "base"
    file_ext = ""
    parser_desc = ""

    def zip_members(self, z):
        raise NotImplementedError

    def folder_files(self, folder):
        raise NotImplementedError

    def load(self, fileobj):
        """Return an iterable of raw records from an open file object."""
        raise NotImplementedError

    def group_games(self, records, fname):
        """Split ``records`` into per-game dicts with ``events``/``team``/``game``."""
        raise NotImplementedError


class JsonSource(LogSource):
    name = "json"
    file_ext = ".json"
    parser_desc = "rowify-jsons"

    def zip_members(self, z):
        # ``*.phases.json`` sidecars (run_benchmark --save-phases) are lists of phase
        # STRINGS, not event logs -- match game event logs only.
        return [
            name
            for name in z.namelist()
            if "game" in name
            and name.endswith(".json")
            and not name.endswith(".phases.json")
        ]

    def folder_files(self, folder):
        return [
            f
            for f in glob.glob(os.path.join(folder, "game*.json"))
            if not f.endswith(".phases.json")
        ]

    def load(self, fileobj):
        return json.load(fileobj)

    def game_digest(self, objs):
        start = None
        for o in objs:
            if o.get("event", "STATE") == "STARTGAME":
                start = o.get("game", dict())
                break
        if start is None:
            start = objs[0].get("game", dict())
        return _digest_start_state(start)

    def group_games(self, records, fname):
        count = 0
        results = [{"events": []}]
        for o in records:
            results[count]["events"].append(o)
            if o.get("event", "STATE") == "POSTGAME":
                results.append({"events": []})
                count += 1

        for g in results[:count]:
            g["team"] = team_fixed(g["events"], fname)
            g["game"] = self.game_digest(g["events"])
        return results[:count]


class MsgpackSource(LogSource):
    name = "msgpack"
    file_ext = ".bin"
    parser_desc = "rowify-bins"

    def zip_members(self, z):
        return [name for name in z.namelist() if name.endswith(".bin")]

    def folder_files(self, folder):
        return glob.glob(os.path.join(folder, "*.bin"))

    def load(self, fileobj):
        import msgpack  # lazy: the JSON path needs no msgpack

        return msgpack.Unpacker(fileobj)

    def open_binary(self):
        return True

    @staticmethod
    def _combo_damage(game, sel_cards):
        """Immunity-adjusted attack damage of ``sel_cards``, straight from the engine.

        Builds the real ``Combo`` from the selected cards' location bitmask (via the
        ``combomap`` bijection) and asks the engine for the damage -- this accounts for
        CLUBS_DOUBLE / SPADES / JOKER_NERF exactly as gameplay does, replacing the old
        by-hand multiplier that sniffed the clubs glyph out of each card's display string.
        Falls back to the plain strength sum when there is no enemy or the card set has
        no attack cell (both cases the old multiplier treated as ``x1``).
        """
        if not sel_cards:
            return 0
        base = sum(x.strength for x in sel_cards)
        if len(game.enemy_pile) == 0:
            return base
        from regi_py.core import ComboTable, cards_bitwise
        from regi_py.combomap import cell_of_bitwise

        cell = cell_of_bitwise(cards_bitwise(sel_cards))
        if cell is None:
            return base
        combo = ComboTable.make_combo(*cell)
        return game.get_combo_damage(game.enemy_pile[0], combo)

    def phase_str_to_game_dct(self, info):
        from regi_py.core import PhaseInfo, GameState, RandomStrategy
        from regi_py.logging import DummyLog
        from regi_py.logging.utils import dump_debug, dump_card

        phase = PhaseInfo.from_string(info["phase"])
        event = "ATTACK" if phase.phase_attacking else "DEFEND"
        game = GameState(DummyLog())
        strat = RandomStrategy()
        for _ in range(phase.num_players):
            game.add_player(strat)
        game.init_phaseinfo(phase)
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
            selected_ind = info["sel_index"]
            selected_combo = info["combos"][selected_ind]
            sel_cards = [x for x in acp.cards if str(x) in selected_combo]
            cmb = [dump_card(x) for x in sel_cards]
            r["combo"] = cmb
            if event == "ATTACK":
                dmg = self._combo_damage(game, sel_cards)
                data["used_combos"].append(cmb)
        if event == "ATTACK":
            r["damage"] = dmg
        else:
            r["maxblock"] = sum(x.strength for x in acp.cards)
        r["last"] = phase.game_endvalue != 0
        return r

    def game_digest(self, objs):
        start = objs[0].get("game", dict())
        digest = _digest_start_state(start)
        # postprocess reconstructed game data (phase index + known strategy)
        for i, o in enumerate(objs):
            o["game"]["phase_count"] = i
            o["game"]["active_player"]["strategy"] = "mcts-explorer"
            for p in o["game"]["players"]:
                p["strategy"] = "mcts-explorer"
        return digest

    def group_games(self, records, fname):
        count = 0
        results = [{"events": []}]
        for o in records:
            try:
                r = self.phase_str_to_game_dct(o)
                results[count]["events"].append(r)
                if r["last"]:
                    results.append({"events": []})
                    count += 1
            except Exception as e:
                print("state loading error:", e)

        for g in results[:count]:
            g["game"] = self.game_digest(g["events"])
            g["team"] = team_fixed(g["events"], fname)
        return results[:count]


# --------------------------------------------------------------------------- #
# shared driver
# --------------------------------------------------------------------------- #
def proc_file(fname, source, run_id, z=None):
    print(f"processing {fname}")
    bname, sim = get_metas(fname)
    binary = getattr(source, "open_binary", lambda: False)()
    try:
        if z is not None:
            fileobj = z.open(fname, "rb") if binary else z.open(fname)
        else:
            fileobj = open(fname, "rb") if binary else open(fname)
        records = source.load(fileobj)
    except Exception as e:
        print("skipped", fname, e)
        return []

    games = source.group_games(records, bname)
    rows = []
    for g in games:
        team = g["team"]
        game = g["game"]
        for e in g["events"]:
            if e.get("event", "STATE") in IGNORE_EVENTS:
                continue
            rows.append(proc_event(e, run_id, bname, game, team, sim))
    return rows


def discover_files(input_object, source):
    if input_object.endswith(".zip"):
        z = zipfile.ZipFile(input_object, "r")
        return z, source.zip_members(z)
    if input_object.endswith(source.file_ext):
        return None, [input_object]
    return None, source.folder_files(input_object)


# --------------------------------------------------------------------------- #
# output sinks: a CSV file and/or a SQLite table, both fed the same rows
# --------------------------------------------------------------------------- #
class CsvSink:
    """Write rows to a CSV, header first (the historical R-facing output)."""

    def __init__(self, path, header):
        self._f = open(path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow(header)

    def write_rows(self, rows):
        self._writer.writerows(rows)

    def close(self):
        self._f.close()


class SqliteSink:
    """Write rows to a ``game_logs`` table in a SQLite database.

    The columns are exactly the CSV header (dotted names quoted), so the DB
    carries the same schema as the CSV -- one row per event, all values stored
    with TEXT affinity (SQLite keeps ints/NULLs as-is regardless).  An existing
    table is dropped first so re-running overwrites rather than appends.
    """

    TABLE = "game_logs"

    def __init__(self, path, header):
        self._conn = sqlite3.connect(path)
        cols_ddl = ", ".join(f'"{c}" TEXT' for c in header)
        self._conn.execute(f'DROP TABLE IF EXISTS "{self.TABLE}"')
        self._conn.execute(f'CREATE TABLE "{self.TABLE}" ({cols_ddl})')
        placeholders = ", ".join("?" * len(header))
        self._insert = f'INSERT INTO "{self.TABLE}" VALUES ({placeholders})'

    def write_rows(self, rows):
        self._conn.executemany(self._insert, rows)

    def close(self):
        self._conn.commit()
        self._conn.close()


def write_outputs(files, source, run_id, output_csv=None, output_db=None, z=None):
    """Parse each file once and fan its rows out to every requested sink."""
    header = FILEMETA + COLNAMES + PLAYERINFO
    sinks = []
    if output_csv:
        sinks.append(CsvSink(output_csv, header))
    if output_db:
        sinks.append(SqliteSink(output_db, header))
    try:
        for file in files:
            rows = proc_file(file, source, run_id, z)
            for sink in sinks:
                sink.write_rows(rows)
    finally:
        for sink in sinks:
            sink.close()


def write_csv(files, source, output_csv, run_id="", z=None):
    """Back-compat wrapper: CSV-only output (delegates to :func:`write_outputs`)."""
    write_outputs(files, source, run_id, output_csv=output_csv, z=z)


SOURCES = {"json": JsonSource, "msgpack": MsgpackSource}


def main(source=None):
    if source is None:
        parser = argparse.ArgumentParser("logs2df")
        parser.add_argument(
            "-s",
            "--source",
            choices=sorted(SOURCES),
            required=True,
            help="log format: json event logs or msgpack MCTS records",
        )
    else:
        if isinstance(source, str):
            source = SOURCES[source]()
        parser = argparse.ArgumentParser(source.parser_desc)
    parser.add_argument(
        "-i",
        "--input-object",
        required=True,
        help="a folder, single log file, or ZIP archive of logs",
    )
    parser.add_argument("-o", "--output-csv", default=None, help="output csv")
    parser.add_argument(
        "-d",
        "--sqlite",
        dest="output_db",
        default=None,
        help="output SQLite database (rows go into a 'game_logs' table)",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.date.today().isoformat(),
        help="value for the 'run_id' column tagging every row "
        "(e.g. r1/r2; default: today's date yyyy-mm-dd)",
    )
    d = parser.parse_args()

    if not d.output_csv and not d.output_db:
        parser.error("need an output: -o/--output-csv and/or -d/--sqlite")

    if source is None:
        source = SOURCES[d.source]()

    z, files = discover_files(d.input_object, source)
    write_outputs(
        files, source, d.run_id, output_csv=d.output_csv, output_db=d.output_db, z=z
    )


if __name__ == "__main__":
    main()
