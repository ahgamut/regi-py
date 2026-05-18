import argparse
import os
import random
import sys
import time
import traceback

#
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

#
from regi_py.core import PhaseInfo
from regi_py import GameState, DummyLog, CXXConsoleLog
from regi_py import get_strategy_map
from regi_py.strats import RandomStrategy
from regi_py.rl import (
    BasicNet,
    PUCTExplorerStrategy,
    NetDirectStrategy,
    KeepyPUCTNode,
    PUCTDataset,
    PUCTDataLoader,
)


def run_epoch(model, data, optimizer):
    y_hat, v_hat = model(data)
    loss = model.calculate_loss(data["y"], data["value"], y_hat, v_hat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def total_enemy_hp(game):
    return sum(x.hp for x in game.enemy_pile)


class EndGameLog(DummyLog):
    def __init__(self):
        super().__init__()
        self.e0 = 0
        self.e1 = 0
        self.reason = None

    def startgame(self, game):
        self.e0 = total_enemy_hp(game)

    def endgame(self, reason, game):
        self.reason = reason
        self.e1 = total_enemy_hp(game)

    def diffe(self):
        return f"{self.e0-self.e1}({self.reason.value})"


def test_model(episode, model, num_simulations):
    model.eval()
    log = EndGameLog()
    diffe = []
    for s in range(10):
        game = GameState(log)
        num_players = random.randint(2, 4)
        for i in range(num_players):
            game.add_player(PUCTExplorerStrategy(model))
        game.initialize()
        e0 = total_enemy_hp(game)
        game.start_loop()
        e1 = total_enemy_hp(game)
        diffe.append(log.diffe())
    print("test games:", diffe, file=sys.stderr)
    torch.save(model.state_dict(), f"./weights/model_{model.__mname__}_{episode}.pt")
    print("episode", episode, "saved model", file=sys.stderr)


def improved_gameplay(episode, new_model, old_model, num_simulations, threshold=0.6):
    new_model.eval()
    old_model.eval()
    log1 = EndGameLog()
    log2 = EndGameLog()

    newer_better = 0
    old_strat = NetDirectStrategy(old_model)
    new_strat = NetDirectStrategy(new_model)

    for s in range(num_simulations):
        game1 = GameState(log1)
        game2 = GameState(log2)
        #
        num_players = random.randint(2, 4)
        for i in range(num_players):
            game1.add_player(old_strat)
            game2.add_player(new_strat)
        game1.initialize()
        game2._init_phaseinfo(game1.export_phaseinfo())
        #
        game1.start_loop()
        game2.start_loop()
        #
        diff1 = log1.e0 - log1.e1
        diff2 = log2.e0 - log2.e1
        #
        if diff2 > diff1 and diff2 != 0:
            print(f"{s} old: {diff1}, new: {diff2} => new is better")
            newer_better += 1

    nb_ratio = newer_better / num_simulations
    print(f"{episode} newer better in {100*nb_ratio:.4f}% of games", file=sys.stderr)
    return nb_ratio > threshold


def get_split_optimizer(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "batchnorm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    grps = [
        {"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0},
    ]

    optimizer = torch.optim.AdamW(grps, lr=5e-3)
    return optimizer


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


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
    dataset = PUCTDataset(maxsize=params.memory_size)
    loader = PUCTDataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=0,
    )
    while ep < params.num_episodes:
        if queue.qsize() >= 1:
            try:
                dataset.add_game(bench_model, queue.get())
            except Exception as e:
                print(f"P{tid} error loading sample:", e)
                traceback.print_exc()
                continue
        if len(dataset) < params.memory_size:
            time.sleep(1)
            continue

        losses = []
        ldr = infinite(loader)
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
        if improved_gameplay(
            ep,
            new_model=bench_model,
            old_model=shared_model,
            num_simulations=16,
            threshold=0.1,
        ):
            print("episode", ep, "updated model", file=sys.stderr)
            shared_model.load_state_dict(train_model.state_dict())
            # test_model(ep, shared_model, params.num_simulations)
            ep += 1

    torch.save(shared_model.state_dict(), f"./weights/model_{model.__mname__}_end.pt")


def simulate_node(root_node, iterations):
    for i in range(iterations):
        node = KeepyPUCTNode.select(root_node)
        if not node.is_terminal():
            node = node.expand()
        reward = node.simulate()
        KeepyPUCTNode.update(node, reward)


def run_single_game(tid, i, net, num_bots, num_iterations):
    a = time.time()
    log = DummyLog()
    strat = RandomStrategy()
    game = GameState(log)
    for _ in range(num_bots):
        game.add_player(strat)
    game.initialize()
    start_phase = game.export_phaseinfo()
    #
    history = []
    node = KeepyPUCTNode(start_phase, net=net, prior=1.0, trim=True, weight=1.414)
    s0 = sum(max(x.hp, 0) for x in node.root_phase.enemy_pile)
    while node.root_phase.game_endvalue == 0:
        simulate_node(node, num_iterations)
        history.append(node.export())
        child = node.best_child_node
        child.parent = None
        node = child
    history.append(node.export())
    win = float(node.root_phase.game_endvalue == 1)
    s1 = sum(max(x.hp, 0) for x in node.root_phase.enemy_pile)
    #
    dt = time.time() - a
    diff = (360 - s1) / 360
    for info in history:
        info.value = diff
    # print(f"{tid},{i},p{len(history)},{s0},{s1},{dt:.4f}s,{win}", file=sys.stderr)
    return history


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
            if len(examples) > 1:
                queue.put(examples)
                count += 1
        except Exception as e:
            print(tid, "unable to explore game", count)
            # traceback.print_exc()


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
