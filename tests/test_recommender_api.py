"""Recommender API contract (torch-free): ``getRecommendedMoves`` returns a ranked
list of ``Combo`` objects, not strings.

This guards the unification used by the webapp recommender panel. Only the
torch-free recommenders (brute, mcts) are exercised here; the NN recommenders
(AZ/ADZ direct + explorer) share the same contract but need torch, so the user
verifies those. Runs wherever the C++ extension is built.
"""
import pytest

pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

from regi_py.core import GameState, RandomStrategy, PhaseInfo, Combo  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402
from regi_py.strats import BruteSamplingStrategy  # noqa: E402
from regi_py.strats.mcts_explorer import MCTSExplorerStrategy  # noqa: E402
from regi_py.strats.phase_utils import PhaseExpander  # noqa: E402


def _collect_phases(seed, n_games=8):
    import regi_py.core as core

    core.seed(seed)
    attack, defense = [], []
    for _ in range(n_games):
        game = GameState(DummyLog())
        for _ in range(2):
            game.add_player(RandomStrategy())
        game.initialize()
        game.start_loop()
        for ph in game.history:  # copy via to_string while the game is alive
            (attack if ph.phase_attacking else defense).append(ph.to_string())
    return attack, defense


def _check_recos(strat, phase_strs, limit=20):
    checked = 0
    for s in phase_strs[:limit]:
        phase = PhaseInfo.from_string(s)
        combos = PhaseExpander(phase).offered()
        if not combos:
            continue
        offered_bw = {c.bitwise for c in combos}
        recos = strat.getRecommendedMoves(phase, combos)
        assert isinstance(recos, list)
        assert len(recos) > 0
        assert len(recos) <= strat.num_recos
        for combo in recos:
            # the unified contract: Combo objects, drawn from the offered set
            assert isinstance(combo, Combo), f"{type(combo)} is not a Combo"
            assert combo.bitwise in offered_bw
        checked += 1
    return checked


def test_brute_recommender_returns_combos(seeded):
    attack, defense = _collect_phases(seeded)
    assert attack and defense
    strat = BruteSamplingStrategy(iterations=16)
    assert _check_recos(strat, attack) > 0
    assert _check_recos(strat, defense) > 0


def test_mcts_recommender_returns_combos(seeded):
    attack, defense = _collect_phases(seeded)
    assert attack and defense
    strat = MCTSExplorerStrategy(iterations=16)
    assert _check_recos(strat, attack) > 0
    assert _check_recos(strat, defense) > 0
