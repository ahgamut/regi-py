"""Shape/contract smoke tests for the pluggable net architectures.

Torch-guarded (skips where torch isn't installed, matching the suite convention).
Parametrized over every registered net, so new architectures are covered the
moment they are added to the registry.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from regi_py.core import (  # noqa: E402
    GameState,
    RandomStrategy,
    MAX_CARDS_IN_GAME,
    MAX_PLAYED_STATUS,
)
from regi_py.logging import DummyLog  # noqa: E402
from regi_py.rl.nets import net_names, get_net  # noqa: E402
from regi_py.rl.az_explorer import AlphaZeroNode, AZNodeInfo  # noqa: E402
from regi_py.rl.training import run_epoch, get_split_optimizer  # noqa: E402


def _fresh_window(net_cls):
    """A real ``max_history`` phase window from a freshly initialized game."""
    game = GameState(DummyLog())
    for _ in range(2):
        game.add_player(RandomStrategy())
    game.initialize()
    phase = game.export_phaseinfo()
    return AlphaZeroNode._trimmed_history([], phase, net_cls.max_history)


@pytest.mark.parametrize("name", net_names())
def test_forward_predict_train_contract(name, seeded):
    net_cls = get_net(name)
    net = net_cls()
    net.eval()
    hist = _fresh_window(net_cls)

    # --- forward on a single tensorized window (no_grad: shape/range checks only,
    # so float(a.sum()) doesn't warn about converting a grad-tracking tensor) ---
    data = net_cls.tensorify_phases(hist)
    with torch.no_grad():
        v, k, a = net.forward(data)
    assert v.shape == (1, 1)
    assert k.shape == (1, MAX_CARDS_IN_GAME)
    assert a.shape == (1, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)
    assert torch.all(k >= 0) and torch.all(k <= 1)     # sigmoid keepy head
    assert torch.all(v >= -1) and torch.all(v <= 1)    # tanh value head
    assert abs(float(a.sum()) - 1.0) < 1e-4            # masked softmax over the grid

    # --- predict (self-play/eval inference contract: numpy out) ---
    vp, kp, ap = net.predict(hist)
    assert isinstance(vp, float)
    assert kp.shape == (MAX_CARDS_IN_GAME,)
    assert ap.shape == (MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)

    # --- tensorify_training + one optimization step ---
    info = AZNodeInfo(
        history=tuple(hist),
        value=1.0,
        atk_probs=np.zeros((MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS), dtype=np.float32),
        keepyness=np.zeros(MAX_CARDS_IN_GAME, dtype=np.float32),
    )
    batch = net_cls.tensorify_training([info, info])
    for field in net_cls.TRAIN_FIELDS:
        assert field in batch and batch[field].shape[0] == 2

    net.train()
    opt = get_split_optimizer(net)
    batch_tuple = tuple(batch[f] for f in net_cls.TRAIN_FIELDS)
    loss, comps = run_epoch(net, batch_tuple, opt)
    assert np.isfinite(loss)
    assert all(np.isfinite(c) for c in comps)
