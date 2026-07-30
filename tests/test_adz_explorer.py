"""Torch-free tests for the ADZ node/explorer stack (``rl/adz/explorer.py``).

``adz.explorer`` itself is torch-free (core + strats + ``rl.features``/``rl.utils``,
all numpy), but importing the ``regi_py.rl`` package runs its ``__init__`` which
pulls in the torch-dependent nets. To keep these runnable wherever the C++
extension is built (torch NOT required), we shadow ``regi_py.rl`` with a torch-free
stub package (same ``__path__``) so its submodules import without the eager net
imports -- the same spirit as ``test_adz_features`` loading ``features.py`` in
isolation. The net is a tiny fake exposing only the ``predict`` contract ADZNode
uses, so the real MCTS/search/export logic is exercised against the real engine.
"""
import sys
import types
import pathlib

import numpy as np
import pytest

pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

# adz_explorer is torch-free, but `from regi_py.rl.features import ...` /
# `from regi_py.rl.utils import *` first run the real rl/__init__, which eagerly
# imports the torch nets. Where torch IS installed that import just works and we use
# it as-is; where it is NOT (the agent env), shadow regi_py.rl with a torch-free stub
# package (same __path__) so its submodules load without the eager net imports. Only
# installed when the real import genuinely fails, so a real rl is never clobbered.
_RL_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "regi_py" / "rl"
if "regi_py.rl" not in sys.modules:
    try:
        import regi_py.rl  # noqa: F401  (real package; succeeds under torch)
    except Exception:
        _stub = types.ModuleType("regi_py.rl")
        _stub.__path__ = [str(_RL_DIR)]
        sys.modules["regi_py.rl"] = _stub

import regi_py.core as core  # noqa: E402
from regi_py import GameState  # noqa: E402
from regi_py.core import RandomStrategy  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402
from regi_py.strats.phase_utils import PhaseExpander  # noqa: E402

import regi_py.rl.adz.explorer as adz  # noqa: E402

MAX_CANDIDATES = 128


class FakeNet:
    """Minimal stand-in for a ``CandidateBaseNet``: uniform priors, fixed value.

    Exercises exactly the inference contract ADZNode/strategies call --
    ``predict(history, offered, phase) -> (value, priors)`` -- and asserts the node
    hands it a full-length history window and the phase's offered combos."""

    __mname__ = "fake"
    max_history = 8
    MAX_CANDIDATES = MAX_CANDIDATES

    def __init__(self, value=0.1):
        self.value = value
        self.max_offered_seen = 0

    def eval(self):
        pass

    def predict(self, history, offered, phase):
        assert len(history) == self.max_history
        k = len(offered)
        self.max_offered_seen = max(self.max_offered_seen, k)
        priors = np.full(k, 1.0 / k, dtype=np.float32) if k else np.zeros(0, np.float32)
        return self.value, priors


def _fresh_game(num_players=2, seed=7):
    core.seed(seed)
    game = GameState(DummyLog())
    for _ in range(num_players):
        game.add_player(RandomStrategy())
    game.initialize()
    return game


def test_trimmed_history_pads_to_maxhist(seeded):
    game = _fresh_game()
    phase = game.export_phaseinfo()
    win = adz.trimmed_history([], phase, 8)
    assert len(win) == 8
    assert win[-1].to_string() == phase.to_string()  # most recent frame is the phase


def test_node_search_and_export_align(seeded):
    net = FakeNet()
    game = _fresh_game()
    root = adz.ADZNode(
        game.export_phaseinfo(), history=list(game.history), net=net, prior=1.0, trim=False
    )
    adz.adz_simulate_node(root, 64)
    info = root.export()

    K = len(info.candidates)
    assert K > 0
    assert info.cand_feats.shape == (K, adz.CAND_FEATURE_DIM)
    assert info.policy.shape == (K,)
    # policy is a visit-fraction distribution over the offered subsets
    assert info.policy.min() >= 0.0
    assert info.policy.sum() <= 1.0 + 1e-6
    # candidates are the offered combos' bitwise identities, aligned to next_combos
    assert info.candidates == [c.bitwise for c in root.next_combos]
    assert isinstance(info, adz.ADZNodeInfo)
    assert info.value == 0.0  # placeholder overwritten by the self-play driver
    assert info.attacking in (0.0, 1.0)
    # search materialized children and backed up visits
    assert len(root.children) > 0
    assert root.visits == 64


def test_terminal_node_export_is_empty():
    # drive a real game to its end, then build a node on the terminal phase
    net = FakeNet()
    game = _fresh_game(seed=3)
    game.start_loop()
    end_phase = game.export_phaseinfo()
    assert end_phase.game_endvalue != 0
    node = adz.ADZNode(end_phase, history=list(game.history), net=net, prior=1.0)
    assert node.is_terminal()
    info = node.export()
    assert len(info.candidates) == 0
    assert info.cand_feats.shape == (0, adz.CAND_FEATURE_DIM)
    # terminal reward, not the net estimate, backs up a terminal leaf
    assert node.simulate() in (1.0,) or node.simulate() <= 0.0


def test_dirichlet_noise_keeps_priors_normalized(seeded):
    net = FakeNet()
    game = _fresh_game()
    root = adz.ADZNode(
        game.export_phaseinfo(), history=list(game.history), net=net, prior=1.0
    )
    if len(root.next_combos) == 0:
        pytest.skip("no offered combos at root")
    root.add_dirichlet_noise()
    assert root.next_priors.min() >= 0.0
    assert abs(float(root.next_priors.sum()) - 1.0) < 1e-5


@pytest.mark.parametrize("num_players", [2, 3])
def test_explorer_strategy_plays_full_game(num_players):
    net = FakeNet()
    core.seed(11 + num_players)
    strat = adz.ADZExplorerStrategy(net, iterations=16)
    game = GameState(DummyLog())
    for _ in range(num_players):
        game.add_player(strat)
    game.initialize()
    game.start_loop()
    # played to a terminal state (win or loss) using valid indices on attack+defense
    assert game.export_phaseinfo().game_endvalue in (1, -1)


def test_direct_strategy_plays_full_game():
    net = FakeNet()
    core.seed(13)
    strat = adz.ADZDirectStrategy(net)
    game = GameState(DummyLog())
    for _ in range(2):
        game.add_player(strat)
    game.initialize()
    game.start_loop()
    assert game.export_phaseinfo().game_endvalue in (1, -1)


def test_offered_candidates_within_pad_width(seeded):
    """Every real decision offers <= MAX_CANDIDATES subsets (the fixed pad width),
    covering both attack and defense nodes."""
    core.seed(1234)
    max_seen = 0
    saw_defense = False
    for _ in range(30):
        g = GameState(DummyLog())
        for _ in range(2):
            g.add_player(RandomStrategy())
        g.initialize()
        g.start_loop()
        for ph in g.history:
            n = len(PhaseExpander(ph).offered())
            max_seen = max(max_seen, n)
            if not ph.phase_attacking and n > 0:
                saw_defense = True
            assert n <= MAX_CANDIDATES
    assert max_seen > 0 and saw_defense
