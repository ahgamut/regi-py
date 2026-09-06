"""The merged log->CSV builder (game_json/logs2df.py).

Locks the R-facing column schema (derived from the serialize tuples but
byte-identical to the historical hardcoded lists), exercises the JSON path
end-to-end on a real JSONLog, and guards the two bugs the merge fixes: the
game_digest ``x0`` pile leak and the recommender ``NotImplemented`` misuse.
"""

import os
import sys

import pytest

from regi_py.core import GameState, RandomStrategy
from regi_py.logging import JSONLog

# game_json/ is a scripts dir, not a package -- put it on the path to import.
_HERE = os.path.dirname(__file__)
_GAME_JSON = os.path.abspath(os.path.join(_HERE, "..", "game_json"))
if _GAME_JSON not in sys.path:
    sys.path.insert(0, _GAME_JSON)

import logs2df  # noqa: E402


# historical, R-facing column order (must stay stable for the analysis scripts)
_HIST_COLNAMES = [
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
_HIST_PLAYERINFO = [
    f"game.players.{pid}.{f}"
    for pid in range(4)
    for f in ("id", "alive", "strategy", "num_cards", "cards")
]


def test_schema_matches_historical_columns():
    assert logs2df.COLNAMES == _HIST_COLNAMES
    assert logs2df.PLAYERINFO == _HIST_PLAYERINFO
    assert logs2df.FILEMETA == ["run_id", "file", "game", "team", "sim"]


def test_playerinfo_field_order_is_a_permutation_of_schema():
    from regi_py.serialize import PLAYER_FIELDS

    assert set(logs2df.PLAYERINFO_FIELD_ORDER) == set(PLAYER_FIELDS)


def test_json_path_end_to_end(tmp_path, seeded):
    fname = tmp_path / "game0-teamR-sim01.json"
    log = JSONLog(str(fname))
    game = GameState(log)
    for _ in range(2):
        game.add_player(RandomStrategy())
    game.initialize()
    game.start_loop()
    log.close()

    rows = logs2df.proc_file(str(fname), logs2df.JsonSource(), "testrun")
    header = logs2df.FILEMETA + logs2df.COLNAMES + logs2df.PLAYERINFO

    assert rows, "expected at least one CSV row from a full game"
    for row in rows:
        assert len(row) == len(header)

    # every row carries the run_id (meta column 0)
    assert {row[0] for row in rows} == {"testrun"}

    event_col = len(logs2df.FILEMETA) + logs2df.COLNAMES.index("event")
    events = {row[event_col] for row in rows}
    # attacks always happen; ignored bookkeeping events are filtered out
    assert "ATTACK" in events
    assert not (events & set(logs2df.IGNORE_EVENTS))

    # the digest (the 'game' meta column) is a stable non-empty hex string per game
    game_col = logs2df.FILEMETA.index("game")
    digests = {row[game_col] for row in rows}
    assert len(digests) == 1 and digests.pop()


def test_sqlite_output_matches_csv(tmp_path, seeded):
    import sqlite3

    fname = tmp_path / "game0-teamR-sim01.json"
    log = JSONLog(str(fname))
    game = GameState(log)
    for _ in range(2):
        game.add_player(RandomStrategy())
    game.initialize()
    game.start_loop()
    log.close()

    header = logs2df.FILEMETA + logs2df.COLNAMES + logs2df.PLAYERINFO
    csv_path = tmp_path / "out.csv"
    db_path = tmp_path / "out.db"
    z, files = logs2df.discover_files(str(fname), logs2df.JsonSource())
    logs2df.write_outputs(
        files,
        logs2df.JsonSource(),
        "testrun",
        output_csv=str(csv_path),
        output_db=str(db_path),
        z=z,
    )

    # the DB carries the same schema and the same rows as the CSV
    conn = sqlite3.connect(str(db_path))
    db_cols = [r[1] for r in conn.execute("PRAGMA table_info(game_logs)")]
    assert db_cols == header
    db_rows = conn.execute(f'SELECT * FROM game_logs').fetchall()
    conn.close()

    csv_rows = logs2df.proc_file(str(fname), logs2df.JsonSource(), "testrun")
    assert len(db_rows) == len(csv_rows) > 0
    # CSV stringifies everything; compare the DB rows the same way so ints/None
    # (stored as-is by SQLite) line up with the CSV's text cells
    def as_csv_text(v):
        return "" if v is None else str(v)

    for db_row, csv_row in zip(db_rows, csv_rows):
        assert [as_csv_text(v) for v in db_row] == [as_csv_text(v) for v in csv_row]


def test_game_digest_x0_fix_uses_piles():
    players = [{"cards": ["Ac", "Kd"]}, {"cards": ["2h"]}]
    with_piles = {
        "players": players,
        "draw_pile": ["3c", "4c"],
        "discard_pile": ["5c"],
        "enemy_pile": ["Js"],
    }
    without_piles = {"players": players}
    # the old x0 bug read piles off the last player (never present) -> the piles
    # were ignored; now they change the digest
    assert logs2df._digest_start_state(with_piles) != logs2df._digest_start_state(
        without_piles
    )


def test_recommender_raises_notimplementederror():
    from regi_py.strats.recommender import RecommenderMixin

    with pytest.raises(NotImplementedError):
        RecommenderMixin().getRecommendedMoves(None, None)
