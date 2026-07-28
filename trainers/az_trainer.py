import argparse
import os
import random
import sys
import time
import traceback

#
import torch
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import DataLoader

from regi_py.rl.basicnet import BasicNet
from regi_py.rl.loaders import ShardBuffer
from regi_py.rl.training import (
    run_epoch,
    run_single_game,
    get_split_optimizer,
    test_model,
    improved_gameplay,
    infinite,
    drain,
)


def trainer(tid, shared_model, queue, train_device, test_device, params):
    print(f"P{tid} on {train_device} to train")
    torch.set_num_threads(params.num_threads)
    with torch.device(train_device):
        train_model = BasicNet()
        train_model.device = train_device
        train_model.load_state_dict(shared_model.state_dict())
        train_model = train_model.to(train_device)
        train_model.train()
        optimizer = get_split_optimizer(train_model)

    with torch.device(test_device):
        bench_model = BasicNet()
        bench_model.device = test_device
        bench_model.load_state_dict(shared_model.state_dict())
        bench_model = bench_model.to(test_device)
        bench_model.eval()

    ep = 0
    buf = ShardBuffer(capacity=params.memory_size)
    while ep < params.num_episodes:
        drain(queue, buf)

        if len(buf) < params.batch_size:
            time.sleep(1)
            continue

        loader = DataLoader(
            dataset=buf.dataset(),
            batch_size=params.batch_size,
            num_workers=1,
        )
        ldr = infinite(loader)
        losses = []
        for e in range(params.epochs):
            batch = next(ldr)
            loss = run_epoch(train_model, batch, optimizer)
            losses.append(loss)

        print(
            "episode",
            ep,
            f"loss={np.mean(losses)}",
            file=sys.stderr,
        )
        bench_model.load_state_dict(train_model.state_dict())
        ep += 1
        if improved_gameplay(
            ep,
            new_model=bench_model,
            old_model=shared_model,
            num_simulations=10,
            threshold=0.5,
        ):
            test_model(ep, shared_model, params.num_simulations)
        shared_model.load_state_dict(train_model.state_dict())

    torch.save(
        shared_model.state_dict(), f"./weights/model_{shared_model.__mname__}_end.pt"
    )


def explorer(tid, shared_model, queue, device, params):
    print(f"P{tid} on {device} to explore")
    torch.set_num_threads(params.num_threads)
    count = 0
    while True:
        num_bots = random.randint(2, 4)
        try:
            examples = run_single_game(
                tid,
                count,
                net=shared_model,
                num_bots=num_bots,
                num_iterations=params.num_simulations,
            )
            queue.put(examples)
            del examples
            count += 1
        except Exception as e:
            print(tid, "unable to explore game", count)
            traceback.print_exc()


def submain(params):
    mp.set_start_method("spawn", force=True)
    #

    if torch.cuda.is_available():
        train_device = "cuda"
    else:
        train_device = "cpu"

    test_device = "cpu"

    with torch.device(test_device):
        shared_model = BasicNet()
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
    processes = []

    p_trainer = mp.Process(
        target=trainer,
        args=(0, shared_model, exp_queue, train_device, test_device, params),
    )
    p_trainer.start()

    for i in range(1, params.num_processes):
        p = mp.Process(
            target=explorer,
            args=(i, shared_model, exp_queue, test_device, params),
        )
        p.start()
        processes.append(p)

    p_trainer.join()
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
        default=4,
        type=int,
        help="number of processes (1 used to train)",
    )
    parser.add_argument(
        "--num-threads", default=1, type=int, help="threads per process"
    )
    parser.add_argument("--test-every", default=1, type=int, help="test every k epochs")
    parser.add_argument("--queue-size", default=64, type=int, help="queue size")
    parser.add_argument("--memory-size", default=64, type=int, help="memory size")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=1, type=int, help="epochs")
    parser.add_argument("--weights-path", default="", help="weights")
    params = parser.parse_args()
    assert params.num_processes >= 2
    if params.num_threads == 0:
        params.num_threads = os.cpu_count() // params.num_processes
        print("setting threads to", params.num_threads)
    submain(params)


if __name__ == "__main__":
    main()
