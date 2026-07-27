"""Strategy, logging, and MCTS/expansion behavior tests.

The get_expansion_at golden-snapshot tests are the key guard for the C11 rewrite:
get_expansion_at is deterministic given the root phase (the legal moves and the
immediate result of playing a chosen combo do not depend on RNG), so a frozen
set of (root phase -> expansion) snapshots must survive the rewrite unchanged.
"""
import gc
import json

import pytest

from regi_py.core import (
    GameState,
    PhaseInfo,
    RandomStrategy,
    DamageStrategy,
    BaseStrategy,
    GameStatus,
)
from regi_py.logging import DummyLog, JSONLog, RegiEncoder
from regi_py.logging.utils import dump_game, dump_debug, dump_player, dump_enemy, dump_card
import regi_py.strats as strats
from regi_py.strats.phase_utils import get_expansion_at, quick_game_value
from regi_py.strats.mcts_explorer import MCTSExplorerStrategy

from conftest import make_game
import _expansion_snapshots as snapshots


# --------------------------------------------------------------------------- #
# strategies play full games
# --------------------------------------------------------------------------- #
# BruteSamplingStrategy is exercised separately (a full game takes seconds).
FAST_STRATS = [c for c in strats.STRATEGY_LIST if c.__name__ != "BruteSamplingStrategy"]


@pytest.mark.parametrize("klass", FAST_STRATS, ids=[c.__name__ for c in FAST_STRATS])
def test_strategy_plays_full_game(klass):
    game = GameState(DummyLog())
    for _ in range(2):
        game.add_player(klass())
    assert game.initialize() == GameStatus.RUNNING
    game.start_loop()
    assert game.status == GameStatus.ENDED


class AlwaysFirstStrategy(BaseStrategy):
    """A minimal Python strategy subclass: always play the first offered combo."""

    __strat_name__ = "always-first"

    def setup(self, player, game):
        return 0

    def getAttackIndex(self, combos, player, yield_allowed, game):
        return 0 if len(combos) else -1

    def getDefenseIndex(self, combos, player, damage, game):
        return 0 if len(combos) else -1

    def getRedirectIndex(self, player, game):
        return (game.active_player + 1) % game.num_players


def test_custom_python_strategy_end_to_end():
    game = GameState(DummyLog())
    for _ in range(3):
        game.add_player(AlwaysFirstStrategy())
    assert game.initialize() == GameStatus.RUNNING
    game.start_loop()
    assert game.status == GameStatus.ENDED


# --------------------------------------------------------------------------- #
# get_expansion_at: golden snapshots (C11 guard) + invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", sorted(snapshots.FIXED_PHASES))
def test_get_expansion_at_matches_golden(label):
    phase = PhaseInfo.from_string(snapshots.FIXED_PHASES[label])
    nxt, combos = get_expansion_at(phase)
    got = [(c.bitwise, p.to_string()) for c, p in zip(combos, nxt)]
    expected = [tuple(pair) for pair in snapshots.EXPECTED[label]]
    assert got == expected


def test_get_expansion_at_is_deterministic():
    phase_str = snapshots.FIXED_PHASES["fresh_attack_2p"]
    first = None
    for _ in range(5):
        nxt, combos = get_expansion_at(PhaseInfo.from_string(phase_str))
        snap = [(c.bitwise, p.to_string()) for c, p in zip(combos, nxt)]
        if first is None:
            first = snap
        else:
            assert snap == first


def test_get_expansion_at_invariants():
    for _ in range(10):
        phase = make_game(2).export_phaseinfo()
        nxt, combos = get_expansion_at(phase)
        assert len(nxt) == len(combos)
        assert all(p is not None for p in nxt)
        for p in nxt:
            s = p.to_string()
            assert PhaseInfo.from_string(s).to_string() == s


def test_quick_game_value_range():
    for _ in range(25):
        phase = make_game(2).export_phaseinfo()
        v = quick_game_value(phase, RandomStrategy)
        assert isinstance(v, (int, float))
        # in-game progress maps to [0, 1]; 2 is the loss sentinel
        assert 0.0 <= v <= 2.0


# --------------------------------------------------------------------------- #
# recommenders (brute + MCTS) return valid moves (tiny iteration budgets)
# --------------------------------------------------------------------------- #
def test_brute_recommender_returns_offered_combo_objects():
    strat = strats.BruteSamplingStrategy(iterations=2, num_recos=5)
    phase = PhaseInfo.from_string(snapshots.FIXED_PHASES["fresh_attack_2p"])
    _, combos = get_expansion_at(phase)
    offered = {c.bitwise for c in combos}
    recos = strat.getRecommendedMoves(phase, combos)
    assert 1 <= len(recos) <= strat.num_recos
    # brute returns Combo objects
    for move in recos:
        assert move.bitwise in offered


def test_mcts_recommender_returns_offered_combo_strings():
    # NOTE: MCTS's getRecommendedMoves returns str(combo), not Combo objects, unlike
    # brute's -- an inconsistency in the recommender contract, characterized here.
    strat = MCTSExplorerStrategy(iterations=3, num_recos=5)
    phase = PhaseInfo.from_string(snapshots.FIXED_PHASES["fresh_attack_2p"])
    _, combos = get_expansion_at(phase)
    offered = {str(c) for c in combos}
    recos = strat.getRecommendedMoves(phase, combos)
    assert 1 <= len(recos) <= strat.num_recos
    for move in recos:
        assert isinstance(move, str)
        assert move in offered


# --------------------------------------------------------------------------- #
# logging + JSON serialization schema
# --------------------------------------------------------------------------- #
def test_dummylog_is_noop_but_complete():
    # DummyLog must be able to drive a whole game with no side effects
    game = make_game(2)
    game.start_loop()
    assert game.status == GameStatus.ENDED


def test_regiencoder_encodes_gamestate_to_valid_json():
    game = make_game(2)
    text = json.dumps(game, cls=RegiEncoder)
    data = json.loads(text)
    assert data["num_players"] == 2
    assert isinstance(data["players"], list) and len(data["players"]) == 2
    assert {"draw_pile", "discard_pile", "enemy_pile"}.issubset(data)


def test_jsonlog_writes_parseable_event_stream(tmp_path):
    path = tmp_path / "game.json"
    log = JSONLog(str(path))
    game = GameState(log)
    for _ in range(2):
        game.add_player(RandomStrategy())
    game.initialize()
    game.start_loop()
    # JSONLog finalizes the array in __del__ (fallback for the legacy
    # create-and-forget usage); the format is a clean array, no {} sentinel.
    del game
    del log
    gc.collect()

    data = json.loads(path.read_text())
    assert isinstance(data, list) and len(data) > 0
    assert {} not in data  # no trailing sentinel object anymore
    kinds = [e["event"] for e in data]
    # dealing emits DRAWONE events before STARTGAME; POSTGAME is the final event.
    assert "STARTGAME" in kinds
    assert kinds[-1] == "POSTGAME"


def test_jsonlog_context_manager_and_empty(tmp_path):
    # explicit close via context manager; an unused log is an empty array
    empty = tmp_path / "empty.json"
    with JSONLog(str(empty)):
        pass
    assert json.loads(empty.read_text()) == []

    used = tmp_path / "used.json"
    with JSONLog(str(used)) as log:
        game = GameState(log)
        for _ in range(2):
            game.add_player(RandomStrategy())
        game.initialize()
        game.start_loop()
    data = json.loads(used.read_text())
    assert isinstance(data, list) and len(data) > 0
    assert data[-1]["event"] == "POSTGAME"


def test_jsonlog_matches_write_json_array(tmp_path):
    # the streamed JSONLog format equals the whole-array writer (one format)
    from regi_py.logging import write_json_array

    streamed = tmp_path / "streamed.json"
    with JSONLog(str(streamed)) as log:
        game = GameState(log)
        for _ in range(2):
            game.add_player(RandomStrategy())
        game.initialize()
        game.start_loop()
    events = json.loads(streamed.read_text())

    whole = tmp_path / "whole.json"
    write_json_array(str(whole), events)
    assert json.loads(whole.read_text()) == events


# Golden schema: lock the dict keys emitted by the dump_* helpers so the C5
# schema unification cannot silently drop or rename a field.
DUMP_GAME_KEYS = {
    "num_players", "active_player_id", "active_player", "phase_count",
    "phase_attacking", "hand_size", "players", "past_yields", "status",
    "used_combos", "current_enemy", "current_block", "progress",
    "draw_pile_size", "discard_pile_size", "enemy_pile_size", "enemy_pile",
}
DUMP_DEBUG_EXTRA_KEYS = {"draw_pile", "discard_pile"}
DUMP_PLAYER_KEYS = {"id", "alive", "num_cards", "strategy", "cards"}
DUMP_ENEMY_KEYS = {"value", "hp", "strength"}
DUMP_CARD_KEYS = {"value", "strength"}


def test_dump_schema_is_stable():
    # use a canonical fresh start so every player has a full hand
    game = GameState(DummyLog())
    for _ in range(2):
        game.add_player(RandomStrategy())
    game.initialize()
    assert set(dump_game(game)) == DUMP_GAME_KEYS
    assert set(dump_debug(game)) == DUMP_GAME_KEYS | DUMP_DEBUG_EXTRA_KEYS
    assert set(dump_player(game.players[0])) == DUMP_PLAYER_KEYS
    assert set(dump_enemy(game.enemy_pile[0])) == DUMP_ENEMY_KEYS
    card = game.players[0].cards[0]
    assert set(dump_card(card)) == DUMP_CARD_KEYS
