"""Correctness harness for the GPU inference server (rl.play_server).

Runs on CPU, no GPU needed. Plays a short self-play game and, for each leaf, checks
the server path reproduces the direct ``net.predict``: (1) the live single-request
path through the arena + a threaded server (exercises predict_remote/exchange), and
(2) a stacked ``predict_batch`` over the collected leaves (exercises B>1 numerics).

  python -m trainers.check_infer_server --net adzmulti --num-simulations 24
"""
import argparse
import sys
import threading
import types

import numpy as np
import torch

from regi_py.rl.play_server import build_arena, infer_server
from regi_py.rl.training import run_single_game
from regi_py.rl.value_fns import get_value_fn
from trainers.trainer import AZ, ADZ


def _flat(out):
    return np.concatenate([np.atleast_1d(np.asarray(x, dtype=np.float64).ravel()) for x in out])


def _tensorify(net_cls, args):
    if len(args) == 1:                       # AZ: (history,)
        return net_cls.tensorify_phases(args[0], None, net_cls.max_history), None
    history, offered, phase = args           # ADZ: (history, offered, phase)
    return net_cls.tensorify_predict(history, offered, phase, None, net_cls.max_history), len(offered)


def _unpack_row(out, k):
    if "priors" in out:                      # ADZ
        return float(out["v"][0]), np.array(out["priors"][:k], dtype=np.float32)
    return (
        float(out["v"][0]),
        np.array(out["k"], dtype=np.float32),
        np.array(out["a"][0], dtype=np.float32),
    )


def main():
    ap = argparse.ArgumentParser("check-infer-server")
    ap.add_argument("--net", default="adzmulti")
    ap.add_argument("--num-simulations", default=24, type=int)
    ap.add_argument("--num-bots", default=3, type=int)
    ap.add_argument("--weights-path", default="")
    ap.add_argument("--atol", default=1e-4, type=float)
    ap.add_argument("--max-checks", default=200, type=int)
    ap.add_argument("--stash", default=64, type=int)
    a = ap.parse_args()

    pl = AZ if a.net in AZ.net_choices else ADZ
    net_cls = pl.get_net(a.net)
    params = types.SimpleNamespace(
        net_cls=net_cls, pipeline=pl, num_threads=1, infer_batch=64,
    )
    value_fn = get_value_fn("hp")

    ref_net = net_cls()
    if a.weights_path:
        ref_net.load_state_dict(torch.load(a.weights_path, weights_only=True, map_location="cpu"))
    ref_net.device = "cpu"
    ref_net.eval()

    arena = build_arena(params, n_slots=1)
    server = threading.Thread(
        target=infer_server, args=(ref_net, arena, "cpu", params), daemon=True
    )
    server.start()

    client_net = net_cls()
    client_net.load_state_dict(ref_net.state_dict())
    client_net.device = "cpu"
    client_net.eval()
    client_net._infer_client = arena.client_for(0)

    worst_live = 0.0
    checks = 0
    stash = []
    orig = ref_net.predict

    def rec(*args):
        nonlocal worst_live, checks
        ref_out = orig(*args)
        if checks < a.max_checks:
            got = client_net.predict(*args)
            worst_live = max(worst_live, float(np.max(np.abs(_flat(ref_out) - _flat(got)))))
            checks += 1
            if len(stash) < a.stash:
                tin, k = _tensorify(net_cls, args)
                stash.append((tin, k, ref_out))
        return ref_out

    ref_net.predict = rec
    run_single_game(0, 0, ref_net, a.num_bots, a.num_simulations, pl.paradigm, value_fn)
    ref_net.predict = orig

    # batched path: stack the stashed leaves into one predict_batch
    worst_batch = 0.0
    if stash:
        keys = list(stash[0][0].keys())
        batch = {kk: torch.cat([s[0][kk] for s in stash], dim=0) for kk in keys}
        out = ref_net.predict_batch(batch)
        for i, (_, k, ref_out) in enumerate(stash):
            row = {f: t[i] for f, t in out.items()}
            got = _unpack_row(row, k)
            worst_batch = max(worst_batch, float(np.max(np.abs(_flat(ref_out) - _flat(got)))))

    print(f"net={a.net} checks={checks} stash={len(stash)}")
    print(f"live (arena+thread server) max|diff| = {worst_live:.3e}")
    print(f"batched predict_batch      max|diff| = {worst_batch:.3e}")
    ok = worst_live < a.atol and worst_batch < a.atol
    print("PASS" if ok else "FAIL", f"(atol={a.atol:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
