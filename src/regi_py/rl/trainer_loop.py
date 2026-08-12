"""Shared multiprocessing orchestration for the NN trainers (AZ + ADZ).

Both pipelines have identical process roles (``trainer`` / ``explorer`` /
``evaluator``) + ``submain`` + CLI, differing only in the net registry and the
paradigm-specific self-play / eval functions. Those differences are bundled in a
:class:`Pipeline` spec that the thin ``trainers/az_trainer.py`` /
``trainers/adz_trainer.py`` shims build and hand to :func:`run_trainer`. The
mp-free plumbing (``run_epoch`` / ``get_split_optimizer`` / ``drain`` /
``ShardBuffer``) is reused from ``rl.training`` unchanged.
"""
import argparse
import gc
import os
import queue
import random
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Sequence

#
import torch
import torch.multiprocessing as mp
import numpy as np

from regi_py.rl.loaders import ShardBuffer
from regi_py.rl.training import (
    run_epoch,
    get_split_optimizer,
    drain,
    run_single_game,
    run_brute_game,
    run_team_game,
    test_model,
    improved_gameplay,
)


@dataclass
class Paradigm:
    """The AZ-vs-ADZ pieces the (otherwise identical) game runners in ``rl.training``
    switch on. The runners duck-type this, so no import back into ``rl.training`` is
    needed; every field is a module-level class/fn, so it pickles across ``spawn``."""

    node_cls: type          # self-play MCTS node (AlphaZeroNode / ADZNode)
    simulate_fn: Callable   # simulate_node / adz_simulate_node
    brute_recorder: type    # RecordingBruteStrategy / RecordingADZBruteStrategy
    team_recorder: type     # RecordingAZTeamStrategy / RecordingADZTeamStrategy
    infos_fn: Callable      # infos_from_game / adz_infos_from_game
    direct_strat: type      # NetDirectStrategy / ADZDirectStrategy (eval: test_model)
    explorer_strat: type    # AZExplorerStrategy / ADZExplorerStrategy (eval: A/B)


@dataclass
class Pipeline:
    """One trainer's paradigm-specific config. ``get_net`` maps ``--net`` to the net
    class; ``paradigm`` bundles the AZ/ADZ classes the shared runners switch on. Stored
    on ``params.pipeline`` and read by ``explorer`` / ``evaluator``; all fields are
    module-level/picklable, so it survives the ``spawn`` hand-off to child processes."""

    prog: str                    # argparse program name
    label: str                   # self-play explorer role label ("az" / "adz")
    net_default: str             # --net default
    net_choices: Sequence[str]   # --net choices
    get_net: Callable            # name -> net_cls
    paradigm: Paradigm           # AZ/ADZ node/strategy/recorder/infos pieces


def trainer(tid, shared_model, exp_queue, eval_queue, eval_done, train_device, params):
    print(f"P{tid} on {train_device} to train")
    torch.set_num_threads(params.num_threads)
    with torch.device(train_device):
        train_model = params.net_cls()
        train_model.device = train_device
        train_model.load_state_dict(shared_model.state_dict())
        train_model = train_model.to(train_device)
        train_model.train()
        optimizer = get_split_optimizer(train_model)

    ep = 0
    buf = ShardBuffer(capacity=params.memory_size, train_fields=params.net_cls.TRAIN_FIELDS)
    part_size = min(5000, params.memory_size // 2)
    while ep < params.num_episodes:
        t0 = time.perf_counter()
        drain(exp_queue, buf)
        t_drain = time.perf_counter() - t0

        if len(buf) < part_size:
            time.sleep(1)
            continue

        losses = []
        comps = []
        t_sample = 0.0
        t_epoch = 0.0
        for e in range(params.epochs):
            t0 = time.perf_counter()
            batch = buf.sample_batch(params.batch_size)
            t1 = time.perf_counter()
            loss, parts = run_epoch(train_model, batch, optimizer)
            t2 = time.perf_counter()
            t_sample += t1 - t0
            t_epoch += t2 - t1
            losses.append(loss)
            comps.append(parts)

        policy, value, keepy = np.mean(comps, axis=0)
        print(
            "episode",
            ep,
            f"loss={np.mean(losses):.4f} policy={policy:.4f} "
            f"value={value:.4f} keepy={keepy:.4f} "
            f"t_drain={t_drain:.3f}s t_sample={t_sample:.3f}s t_epoch={t_epoch:.3f}s",
            file=sys.stderr,
        )
        ep += 1

        # publish the freshest weights so explorers self-play with the latest net
        shared_model.load_state_dict(train_model.state_dict())

        # hand a candidate snapshot to the eval process; best-effort, so a busy
        # evaluator never stalls training (stale candidates are simply skipped)
        if ep % params.test_every == 0:
            # clone: on a CPU trainer state_dict() aliases the live params, so a
            # bare snapshot would keep mutating under the evaluator
            cpu_state = {
                k: v.detach().to("cpu").clone()
                for k, v in train_model.state_dict().items()
            }
            try:
                eval_queue.put_nowait((ep, cpu_state))
            except queue.Full:
                pass

    torch.save(
        shared_model.state_dict(), f"./weights/model_{shared_model.__mname__}_end.pt"
    )
    eval_queue.put(None)  # stop the evaluator
    # a queued candidate is tensors backed by THIS process's shared memory, freed
    # the instant the trainer exits -- which crashes the evaluator's get(). Stay
    # alive until the evaluator has drained everything (it sets eval_done).
    eval_done.wait()


def evaluator(eval_queue, eval_done, params):
    print("Peval to evaluate")
    torch.set_num_threads(params.num_threads)
    pl = params.pipeline
    old_model = params.net_cls()
    old_model.device = "cpu"
    new_model = params.net_cls()
    new_model.device = "cpu"
    have_baseline = False

    try:
        while True:
            item = eval_queue.get()
            if item is None:
                break
            episode, state_dict = item
            new_model.load_state_dict(state_dict)
            new_model.eval()
            # done with the shared-memory tensors; promote from new_model below so
            # nothing references the trainer's shared memory during the eval games
            item = state_dict = None
            # the first candidate becomes the baseline and is always checkpointed
            if not have_baseline:
                old_model.load_state_dict(new_model.state_dict())
                have_baseline = True
                test_model(episode, new_model, params.num_simulations, pl.paradigm)
                continue
            # otherwise only checkpoint when the candidate beats the last-saved one
            if improved_gameplay(
                episode,
                new_model,
                old_model,
                num_simulations=10,
                paradigm=pl.paradigm,
                threshold=0.5,
            ):
                old_model.load_state_dict(new_model.state_dict())
                test_model(episode, new_model, params.num_simulations, pl.paradigm)
    finally:
        # release the trainer (which is holding the shared memory alive for us)
        eval_done.set()


def explorer(tid, shared_model, exp_queue, device, params):
    # odd-tid explorers run net-guided self-play; other_play explorers (even tid)
    # supply the complementary data -- brute late-game samples by default, or full
    # cooperative team games with --team-games (net beside other regi_py.strats).
    pl = params.pipeline
    if params.team_games:
        other_play = (tid % 2) == 0
    else:
        other_play = (tid % 4) == 0
    if other_play:
        role = "team-explore" if params.team_games else "brute-explore"
    else:
        role = f"{pl.label}-explore"
    print(f"P{tid} on {device} to {role}")
    torch.set_num_threads(params.num_threads)
    count = 0
    fails = 0
    logged_gc = False
    while True:
        # reclaim the previous game's search tree (cyclic refs pin native PhaseInfo)
        collected = gc.collect()
        if not logged_gc:
            print(f"P{tid}: gc.collect() reclaimed {collected} objects", file=sys.stderr)
            logged_gc = True
        num_bots = random.randint(2, 4)
        try:
            if other_play:
                if params.team_games:
                    examples = run_team_game(
                        tid,
                        count,
                        shared_model,
                        num_bots,
                        params.num_simulations,
                        pl.paradigm,
                    )
                else:
                    examples = run_brute_game(
                        tid,
                        count,
                        params.net_cls,
                        num_bots,
                        params.num_simulations,
                        pl.paradigm,
                    )
                if examples is None:  # lost/degenerate game -> no data submitted
                    fails = 0
                    continue
            else:
                examples = run_single_game(
                    tid,
                    count,
                    shared_model,
                    num_bots,
                    params.num_simulations,
                    pl.paradigm,
                )
            exp_queue.put(examples)
            del examples
            count += 1
            fails = 0
        except Exception:
            fails += 1
            print(
                f"P{tid} failed to explore game {count} (fail {fails})",
                file=sys.stderr,
            )
            traceback.print_exc()
            if fails >= params.max_explore_fails:
                print(
                    f"P{tid} giving up after {fails} consecutive failures",
                    file=sys.stderr,
                )
                return
            time.sleep(min(fails, 5))  # brief backoff before retrying


def submain(params):
    mp.set_start_method("spawn", force=True)
    #

    if torch.cuda.is_available():
        train_device = "cuda"
    else:
        train_device = "cpu"

    test_device = "cpu"

    with torch.device(test_device):
        shared_model = params.net_cls()
        if os.path.isfile(params.weights_path):
            shared_model.load_state_dict(
                torch.load(
                    params.weights_path, weights_only=True, map_location=test_device
                )
            )
        shared_model.device = test_device
        shared_model.eval()

    shared_model.share_memory()
    exp_queue = mp.Queue(maxsize=params.queue_size)
    eval_queue = mp.Queue(maxsize=2)
    eval_done = mp.Event()  # evaluator -> trainer: "queue drained, safe to exit"
    processes = []

    p_trainer = mp.Process(
        target=trainer,
        args=(0, shared_model, exp_queue, eval_queue, eval_done, train_device, params),
    )
    p_trainer.start()

    p_eval = mp.Process(target=evaluator, args=(eval_queue, eval_done, params))
    p_eval.start()

    for i in range(1, params.num_processes):
        p = mp.Process(
            target=explorer,
            args=(i, shared_model, exp_queue, test_device, params),
        )
        p.start()
        processes.append(p)

    p_trainer.join()
    # on a clean exit the trainer already sent None and waited on eval_done; this
    # best-effort sentinel only matters if the trainer died first, to unblock the
    # evaluator's get()
    try:
        eval_queue.put_nowait(None)
    except queue.Full:
        pass
    p_eval.join()
    for p in processes:
        p.terminate()


def build_parser(pipeline):
    parser = argparse.ArgumentParser(pipeline.prog)
    parser.add_argument(
        "--num-episodes", default=1, type=int, help="number of episodes"
    )
    parser.add_argument(
        "--num-simulations", default=32, type=int, help="number of simulations per game"
    )
    parser.add_argument(
        "--num-processes",
        default=0,
        type=int,
        help="worker procs: 1 trains, the rest explore (eval is a separate proc); "
        "0 = auto-size to fill the CPU",
    )
    parser.add_argument(
        "--num-threads", default=1, type=int, help="threads per process"
    )
    parser.add_argument(
        "--test-every", default=1, type=int, help="offer a candidate to eval every k episodes"
    )
    parser.add_argument(
        "--max-explore-fails",
        default=5,
        type=int,
        help="explorer gives up after this many consecutive game failures",
    )
    parser.add_argument("--queue-size", default=64, type=int, help="queue size")
    parser.add_argument("--memory-size", default=64, type=int, help="memory size")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=1, type=int, help="epochs")
    parser.add_argument("--weights-path", default="", help="weights")
    parser.add_argument(
        "--net",
        default=pipeline.net_default,
        choices=pipeline.net_choices,
        help="net architecture",
    )
    parser.add_argument(
        "--team-games",
        action="store_true",
        help="even-tid explorers play full cooperative games (net beside other "
        "regi_py.strats, training on the NN's decisions only) instead of brute "
        "games; use only after regular self-play has trained the net",
    )
    return parser


def run_trainer(pipeline):
    """Parse args for ``pipeline``, size the process pool, and launch ``submain``.
    The two trainer shims each build a :class:`Pipeline` and call this."""
    params = build_parser(pipeline).parse_args()
    params.pipeline = pipeline
    params.net_cls = pipeline.get_net(params.net)
    ncpu = os.cpu_count() or 2
    if params.num_processes <= 0:
        # fill the CPU with explorers: 1 trainer + (n-1) explorers + 1 evaluator,
        # so n = ncpu - 1 keeps the box saturated without oversubscribing
        params.num_processes = max(2, ncpu - 1)
        print("setting num-processes to", params.num_processes, file=sys.stderr)
    assert params.num_processes >= 2
    if params.num_threads == 0:
        params.num_threads = max(1, ncpu // params.num_processes)
        print("setting threads to", params.num_threads, file=sys.stderr)
    submain(params)
