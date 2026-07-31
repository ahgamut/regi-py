import argparse
import os
import queue
import random
import sys
import time
import traceback

#
import torch
import torch.multiprocessing as mp
import numpy as np

from regi_py.rl.az.nets import get_net, net_names
from regi_py.rl.loaders import ShardBuffer
from regi_py.rl.training import (
    run_epoch,
    run_single_game,
    run_brute_game,
    run_team_game,
    get_split_optimizer,
    test_model,
    improved_gameplay,
    drain,
)


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
        drain(exp_queue, buf)

        if len(buf) < part_size:
            time.sleep(1)
            continue

        losses = []
        comps = []
        for e in range(params.epochs):
            batch = buf.sample_batch(params.batch_size)
            loss, parts = run_epoch(train_model, batch, optimizer)
            losses.append(loss)
            comps.append(parts)

        policy, value, keepy = np.mean(comps, axis=0)
        print(
            "episode",
            ep,
            f"loss={np.mean(losses):.4f} policy={policy:.4f} "
            f"value={value:.4f} keepy={keepy:.4f}",
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
                test_model(episode, new_model, params.num_simulations)
                continue
            # otherwise only checkpoint when the candidate beats the last-saved one
            if improved_gameplay(
                episode,
                new_model=new_model,
                old_model=old_model,
                num_simulations=10,
                threshold=0.5,
            ):
                old_model.load_state_dict(new_model.state_dict())
                test_model(episode, new_model, params.num_simulations)
    finally:
        # release the trainer (which is holding the shared memory alive for us)
        eval_done.set()


def explorer(tid, shared_model, exp_queue, device, params):
    # odd-tid explorers run net-guided AZ self-play; other_play explorers (even tid)
    # supply the complementary data -- brute late-game samples by default, or full
    # cooperative team games with --team-games (net beside other regi_py.strats).
    other_play = (tid % 2) == 0
    if other_play:
        role = "team-explore" if params.team_games else "brute-explore"
    else:
        role = "az-explore"
    print(f"P{tid} on {device} to {role}")
    torch.set_num_threads(params.num_threads)
    count = 0
    fails = 0
    while True:
        num_bots = random.randint(2, 4)
        try:
            if other_play:
                if params.team_games:
                    examples = run_team_game(
                        tid,
                        count,
                        net=shared_model,
                        num_bots=num_bots,
                        iterations=params.num_simulations,
                    )
                else:
                    examples = run_brute_game(
                        tid,
                        count,
                        net_cls=params.net_cls,
                        num_bots=num_bots,
                        iterations=params.num_simulations,
                    )
                if examples is None:  # lost/degenerate game -> no data submitted
                    fails = 0
                    continue
            else:
                examples = run_single_game(
                    tid,
                    count,
                    net=shared_model,
                    num_bots=num_bots,
                    num_iterations=params.num_simulations,
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


def main():
    parser = argparse.ArgumentParser("regi-mcts-trainer")
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
        "--net", default="basic", choices=net_names(), help="net architecture"
    )
    parser.add_argument(
        "--team-games",
        action="store_true",
        help="even-tid explorers play full cooperative games (net beside other "
        "regi_py.strats, training on the NN's decisions only) instead of brute "
        "games; use only after regular self-play has trained the net",
    )
    params = parser.parse_args()
    params.net_cls = get_net(params.net)
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


if __name__ == "__main__":
    main()
