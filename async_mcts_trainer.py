import argparse
import os
import random
import sys
import time

#
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np

#
from regi_py import GameState, DummyLog, CXXConsoleLog
from regi_py import get_strategy_map
from regi_py.rl import BatchedMCTS, MCTS, MC1Model, MCTSTesterStrategy


def MCTSLoss(prob, v, prob_hat, v_hat):
    n = v.shape[0]
    loss1 = nn.functional.mse_loss(v, v_hat)
    # loss2 = nn.functional.mse_loss(prob, prob_hat)
    p1 = prob
    p2 = -torch.log(prob_hat)
    loss2 = (p1 * p2).sum()
    return loss1 + loss2


def run_epoch(model, batch, optimizer):
    states, prob, v = batch
    prob_hat, v_hat = model(states)
    loss = MCTSLoss(prob, v, prob_hat, v_hat)
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
            game.add_player(MCTSTesterStrategy(model))
        game.initialize()
        e0 = total_enemy_hp(game)
        game.start_loop()
        e1 = total_enemy_hp(game)
        diffe.append(log.diffe())
    print("test games:", diffe, file=sys.stderr)
    torch.save(model.state_dict(), f"./weights/model_{episode}.pt")


def trainer(tid, shared_model, queue, device, params):
    print(f"P{tid} on {device} to train")
    torch.set_num_threads(params.num_threads)
    with torch.device(device):
        train_model = MC1Model()
        train_model.device = device
        train_model.load_state_dict(shared_model.state_dict())
        train_model = train_model.to(device)
        train_model.train()
        loss_fn = MCTSLoss
        optimizer = torch.optim.Adam(train_model.parameters(), lr=0.01)

    ep = -1
    samples = []
    while ep < params.num_episodes:
        if queue.qsize() >= params.memory_size:
            samples.clear()
            ep += 1
            try:
                while len(samples) < params.memory_size:
                    samples.append(queue.get())
            except Exception as e:
                print(f"P{tid} error loading sample:", e)
                continue
        elif len(samples) < params.memory_size:
            time.sleep(1)
            continue

        losses = []
        for e in range(params.epochs):
            batch = train_model.tensorify(samples, params.batch_size)
            loss = run_epoch(train_model, batch, optimizer)
            losses.append(loss)

        print("training in episode", ep, "loss =", np.mean(loss), file=sys.stderr)
        shared_model.load_state_dict(train_model.state_dict())
        if ep % params.test_every == 0:
            test_model(ep, shared_model, params.num_simulations)

    torch.save(shared_model.state_dict(), "./weights/model_end.pt")


def explorer(tid, shared_model, queue, device, params):
    print(f"P{tid} on {device} to explore")
    torch.set_num_threads(params.num_threads)
    small_N = params.memory_size // (params.num_processes - 1)
    mcts = BatchedMCTS(
        net=shared_model,
        N=small_N,
        batch_size=params.batch_size,
        randomize=(tid % 2 == 0),
    )
    while True:
        mcts.clear_examples()
        mcts.reset_game()
        mcts.sim_game_full(sims=params.num_simulations)
        examples = mcts.get_examples()
        for x in examples:
            queue.put(x)
        diffe = total_enemy_hp(examples[0].phase) - total_enemy_hp(examples[-1].phase)
        if len(examples) > 0:
            print(
                f"P{tid},q={queue.qsize()},t={len(examples)},e={examples[-1].value},dmg={diffe}",
                file=sys.stderr,
            )


def submain(params):
    mp.set_start_method("spawn", force=True)
    #

    if torch.cuda.is_available():
        train_device = "cuda"
    else:
        train_device = "cpu"

    test_device = "cpu"

    with torch.device(test_device):
        shared_model = MC1Model()
        if os.path.isfile(params.weights_path):
            shared_model.load_state_dict(
                torch.load(
                    params.weights_path, weights_only=True, map_location=test_device
                )
            )
        shared_model.device = test_device
        shared_model.eval()

    shared_model.share_memory()
    exp_queue = mp.Queue(maxsize=params.memory_size)
    processes = []

    p_trainer = mp.Process(
        target=trainer,
        args=(0, shared_model, exp_queue, train_device, params),
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
    parser.add_argument("--memory-size", default=64, type=int, help="memory size")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=1, type=int, help="epochs")
    parser.add_argument(
        "--weights-path", default="./weights/best_model.pt", help="weights"
    )
    params = parser.parse_args()
    assert params.num_processes >= 2
    if params.num_threads == 0:
        params.num_threads = os.cpu_count() // params.num_processes
        print("setting threads to", params.num_threads)
    submain(params)


if __name__ == "__main__":
    main()
