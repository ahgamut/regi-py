"""Profile one training self-play game: where does the time go?

Runs the REAL ``run_single_game`` path for a chosen net + ``--num-simulations`` and
reports the wall-time split across ``net.predict`` (tensorify + forward), the C++
stepping/expansion (``PhaseExpander``), and the remainder (MCTS backup, final
``tensorify_training``, Python overhead) -- plus predict counts and how often the net
INPUT repeats (the achievable hit rate for a leaf-eval / transposition cache, which
must key on the whole history window). ``--cprofile`` adds a function-level breakdown
so you can see tensorify vs forward vs stepping inside ``predict``.

    python -m trainers.profile_selfplay --net adzmulti --num-simulations 1536
    python -m trainers.profile_selfplay --net basic --num-simulations 256 --num-games 3
    python -m trainers.profile_selfplay --net adzmulti --num-simulations 512 --cprofile

Torch env only (imports the nets). Weights are optional -- a random net gives the same
time SPLIT (absolute times scale with net size / weights only via input distribution).
Throwaway diagnostic; not part of the training pipeline.
"""
import argparse
import cProfile
import pstats
import random
import time
from collections import Counter

import torch

from regi_py import seed as core_seed
from regi_py.strats import phase_utils
from regi_py.rl.training import run_single_game
from regi_py.rl.value_fns import get_value_fn
from trainers.trainer import AZ, ADZ


def _select(net_name):
    """The pipeline (AZ/ADZ) whose registry owns ``net_name`` -- same rule as trainer.py."""
    for pl in (AZ, ADZ):
        if net_name in pl.net_choices:
            return pl
    choices = list(AZ.net_choices) + list(ADZ.net_choices)
    raise SystemExit(f"unknown --net {net_name!r}; choices: {choices}")


class Timers:
    def __init__(self):
        self.predict_t = 0.0
        self.predict_n = 0
        self.step_t = 0.0
        self.step_n = 0
        self.window_keys = Counter()  # full net-input window identity (real cache key)
        self.leaf_keys = Counter()    # just the leaf phase (a looser key)


def _instrument(net, timers):
    """Wrap net.predict (instance attr) + PhaseExpander._run (the one C++ stepping entry,
    hit by both offered() and step()) to accumulate times/counts. Returns restore()."""
    orig_predict = net.predict
    orig_run = phase_utils.PhaseExpander._run

    def timed_predict(history, *a, **k):
        timers.window_keys["|".join(p.to_string() for p in history)] += 1
        timers.leaf_keys[history[-1].to_string()] += 1
        t = time.perf_counter()
        try:
            return orig_predict(history, *a, **k)
        finally:
            timers.predict_t += time.perf_counter() - t
            timers.predict_n += 1

    def timed_run(self, *a, **k):
        t = time.perf_counter()
        try:
            return orig_run(self, *a, **k)
        finally:
            timers.step_t += time.perf_counter() - t
            timers.step_n += 1

    net.predict = timed_predict
    phase_utils.PhaseExpander._run = timed_run

    def restore():
        net.predict = orig_predict
        phase_utils.PhaseExpander._run = orig_run

    return restore


def _report(timers, total, args, num_bots):
    n = max(1, timers.predict_n)
    step_n = max(1, timers.step_n)
    other = total - timers.predict_t - timers.step_t
    pct = lambda x: 100.0 * x / total if total else 0.0
    print("\n=== self-play profile ===")
    print(
        f"net={args.net} sims={args.num_simulations} players={num_bots} "
        f"games={args.num_games} threads={args.num_threads}"
    )
    print(f"total                 : {total:9.2f}s   ({total / args.num_games:.2f}s/game)")
    print(
        f"net.predict           : {timers.predict_t:9.2f}s  ({pct(timers.predict_t):5.1f}%)  "
        f"n={timers.predict_n}  {1e3 * timers.predict_t / n:.2f} ms/call"
    )
    print(
        f"C++ stepping (_run)   : {timers.step_t:9.2f}s  ({pct(timers.step_t):5.1f}%)  "
        f"n={timers.step_n}  {1e3 * timers.step_t / step_n:.3f} ms/call"
    )
    print(f"other (backup/tensorify/py): {other:9.2f}s  ({pct(other):5.1f}%)")
    uw, ul = len(timers.window_keys), len(timers.leaf_keys)
    print(
        f"predict inputs        : {timers.predict_n} calls, "
        f"{uw} distinct windows -> {100 * (1 - uw / n):.1f}% repeat (cache hit ceiling), "
        f"{ul} distinct leaf-phases -> {100 * (1 - ul / n):.1f}% repeat"
    )
    print(
        "  (a leaf-eval cache must key on the WHOLE window, so 'distinct windows' is the\n"
        "   achievable hit rate; leaf-phase repeat is only an upper bound. Rates span all\n"
        f"   {args.num_games} game(s) -- a within-game/persistent cache sees at most this.)"
    )


def main():
    ap = argparse.ArgumentParser("profile-selfplay")
    ap.add_argument("--net", default="basic")
    ap.add_argument("--num-simulations", type=int, default=256)
    ap.add_argument("--num-games", type=int, default=1)
    ap.add_argument("--num-bots", type=int, default=0, help="players; 0 = random 2-4 (like training)")
    ap.add_argument("--weights-path", default="")
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cprofile", action="store_true", help="also dump a function-level breakdown")
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    pipeline = _select(args.net)
    net = pipeline.get_net(args.net)()
    net.device = "cpu"
    if args.weights_path:
        net.load_state_dict(
            torch.load(args.weights_path, weights_only=True, map_location="cpu")
        )
    net.eval()

    value_fn = get_value_fn("hp")
    timers = Timers()
    restore = _instrument(net, timers)

    def run_all():
        for g in range(args.num_games):
            core_seed(args.seed + g)
            random.seed(args.seed + g)
            num_bots = args.num_bots or random.randint(2, 4)
            run_single_game(0, g, net, num_bots, args.num_simulations, pipeline.paradigm, value_fn)
            run_all.last_bots = num_bots

    t0 = time.perf_counter()
    if args.cprofile:
        pr = cProfile.Profile()
        pr.enable()
        run_all()
        pr.disable()
    else:
        run_all()
    total = time.perf_counter() - t0
    restore()

    _report(timers, total, args, getattr(run_all, "last_bots", args.num_bots))

    if args.cprofile:
        print("\n=== cProfile (top 30 by cumulative time) ===")
        print("NOTE: cProfile adds overhead -- read the SPLIT from a run WITHOUT --cprofile;")
        print("use this only for the tensorify-vs-forward-vs-step function breakdown.")
        pstats.Stats(pr).sort_stats("cumulative").print_stats(30)


if __name__ == "__main__":
    main()
