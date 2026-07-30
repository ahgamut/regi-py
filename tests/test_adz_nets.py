"""Shape/contract smoke tests for the ADZ candidate-scoring nets.

Torch-guarded (skips where torch isn't installed, matching the suite convention).
Parametrized over every registered ADZ net, so new candidate encodings are covered
the moment they join the registry. Builds a real decision + offered list from a
freshly initialized game and checks the ``forward`` / ``predict`` /
``tensorify_training`` / ``run_epoch`` contract from ``CandidateBaseNet``.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from regi_py.core import GameState, RandomStrategy, MAX_CARDS_IN_GAME  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402
from regi_py.strats.phase_utils import PhaseExpander  # noqa: E402
from regi_py.rl.features import candidate_semantics  # noqa: E402
from regi_py.rl.adz_nets import adz_net_names, get_adz_net  # noqa: E402
from regi_py.rl.adz_explorer import ADZNode, ADZNodeInfo, trimmed_history  # noqa: E402
from regi_py.rl.training import run_epoch, get_split_optimizer  # noqa: E402


def _decision(net_cls):
    """A real decision: (history window, offered combos, phase) from a fresh game."""
    game = GameState(DummyLog())
    for _ in range(2):
        game.add_player(RandomStrategy())
    game.initialize()
    phase = game.export_phaseinfo()
    offered = PhaseExpander(phase).offered()
    history = trimmed_history([], phase, net_cls.max_history)
    return history, offered, phase


def _info(net_cls, history, offered, phase, value=1.0):
    """An ADZNodeInfo with a normalized visit-fraction policy over the offer."""
    K = len(offered)
    policy = np.full(K, 1.0 / K, dtype=np.float32) if K else np.zeros(0, np.float32)
    return ADZNodeInfo(
        history=tuple(history),
        candidates=[c.bitwise for c in offered],
        cand_feats=candidate_semantics(phase, offered),
        policy=policy,
        value=value,
        attacking=float(phase.phase_attacking),
    )


@pytest.mark.parametrize("name", adz_net_names())
def test_forward_masks_and_sums_over_real_candidates(name, seeded):
    net_cls = get_adz_net(name)
    net = net_cls()
    net.eval()
    history, offered, phase = _decision(net_cls)
    K = len(offered)
    assert K > 0

    data = net_cls.tensorify_predict(history, offered, phase)
    with torch.no_grad():
        value, cand_logits, keepy = net.forward(data)
    assert value.shape == (1, 1)
    assert cand_logits.shape == (1, net_cls.MAX_CANDIDATES)
    assert keepy.shape == (1, MAX_CARDS_IN_GAME)
    assert torch.all(value >= -1) and torch.all(value <= 1)   # tanh value head
    assert torch.all(keepy >= 0) and torch.all(keepy <= 1)    # sigmoid keepy head
    # padded slots are -inf; a softmax over the whole row is a distribution over the
    # real candidates only, and sums to 1
    assert torch.isneginf(cand_logits[0, K:]).all()
    probs = torch.softmax(cand_logits, dim=-1)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert abs(float(probs[0, :K].sum()) - 1.0) < 1e-4


@pytest.mark.parametrize("name", adz_net_names())
def test_predict_returns_aligned_priors(name, seeded):
    net_cls = get_adz_net(name)
    net = net_cls()
    net.eval()
    history, offered, phase = _decision(net_cls)

    v_hat, priors = net.predict(history, offered, phase)
    assert isinstance(v_hat, float)
    assert priors.shape == (len(offered),)
    assert abs(float(priors.sum()) - 1.0) < 1e-4
    # empty offer -> empty priors, no crash
    v0, p0 = net.predict(history, [], phase)
    assert p0.shape == (0,)


@pytest.mark.parametrize("name", adz_net_names())
def test_tensorify_training_and_one_epoch(name, seeded):
    net_cls = get_adz_net(name)
    net = net_cls()
    history, offered, phase = _decision(net_cls)
    info = _info(net_cls, history, offered, phase)

    batch = net_cls.tensorify_training([info, info])
    for field in net_cls.TRAIN_FIELDS:
        assert field in batch and batch[field].shape[0] == 2
    # policy rows are normalized visit fractions padded to MAX_CANDIDATES
    assert batch["policy"].shape == (2, net_cls.MAX_CANDIDATES)
    assert abs(float(batch["policy"][0].sum()) - 1.0) < 1e-4

    net.train()
    opt = get_split_optimizer(net)
    batch_tuple = tuple(batch[f] for f in net_cls.TRAIN_FIELDS)
    loss, comps = run_epoch(net, batch_tuple, opt)
    assert np.isfinite(loss)
    assert len(comps) == 3 and all(np.isfinite(c) for c in comps)
