import argparse
import os
import random
import sys

#
import torch
import torch.nn as nn
import numpy as np

#
from regi_py import GameState
from regi_py import get_strategy_map
from regi_py.rl import MemoryLog, RL1Model

BEST_REMAIN = 350
STRATEGY_MAP = get_strategy_map()


def set_rewards(log, start, end, model, epsilon):
    first_state = log.memories[start]
    last_state = log.memories[end - 1]
    global BEST_REMAIN
    last_state["reward"] = (360 - last_state["remaining"]) / 72
    last_state["best_future"] = last_state["reward"]

    progress = 360 - (first_state["remaining"] - last_state["remaining"])
    if progress <= 1.1 * BEST_REMAIN and epsilon <= 0.2:
        if progress <= BEST_REMAIN:
            BEST_REMAIN = progress
            print("new best!", BEST_REMAIN)
            torch.save(model.state_dict(), f"./weights/model_{BEST_REMAIN}.pt")
            BEST_REMAIN *= 0.95
        last_state["best_future"] *= 2
    last_state["best_future"] = min(20, last_state["best_future"])

    for i in range((end - 2), (start - 1), -1):
        cur_state = log.memories[i]
        next_state = log.memories[i + 1]
        if cur_state["attacking"]:
            diff = cur_state["remaining"] - next_state["remaining"]
            if diff == cur_state["proxy"]:
                diff += 20  # reward exact kills
            if diff == 0:  # avoid yields
                diff = -30
            cur_state["reward"] = diff
        cur_state["best_future"] = (
            cur_state["reward"] / 72 + 0.99 * model.predict(next_state).max().item()
        )
        cur_state["best_future"] = max(-10, cur_state["best_future"])

    for j in range(start, end):
        jj = j - start
        state = log.memories[j]
        # print(jj, state["attacking"], state["reward"], state["best_future"])


def basic_game(strats, log, model, epsilon, collect=True):
    n_players = len(strats)
    assert n_players in [2, 3, 4], "only 2, 3, or 4 players"

    print(strats, end=" ", file=sys.stderr)
    start = len(log.memories)
    game = GameState(log)
    for i in range(n_players):
        cls = STRATEGY_MAP[strats[i]]
        if collect:
            cls = log.record(cls)
        obj = cls()
        if strats[i] == "rl1":
            obj.model = model
            obj.epsilon = epsilon
        game.add_player(obj)
    game._init_random()
    game.start_loop()
    end = len(log.memories)
    set_rewards(log, start, end, model, epsilon)


def run_epoch(model, batch, optimizer, loss_fn, gamma=0.99):
    q_values = model(batch)
    target_q_values = batch["best_future"]
    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def run_episode(log, model, epsilon, collect=True):
    num_teammates = random.randint(1, 2)
    while len(log.memories) <= log.N:
        bots = ["rl1"] * (num_teammates + 1)
        # bots = ["rl1"] + random.sample(list(STRATEGY_MAP.keys()), num_teammates)
        basic_game(bots, log=log, model=model, epsilon=epsilon, collect=collect)


def main():
    parser = argparse.ArgumentParser("regi-rl-trainer")
    parser.add_argument(
        "--num-episodes", default=5, type=int, help="number of episodes"
    )
    parser.add_argument("--memory-size", default=64, type=int, help="memory size")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=1, type=int, help="epochs")
    parser.add_argument(
        "--weights-path", default="./weights/best_model.pt", help="weights"
    )
    d = parser.parse_args()

    log = MemoryLog(N=d.memory_size)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    epsilon = 0.1
    gamma = 0.99

    with torch.device(device):
        model = RL1Model()
        model.device = device
        if os.path.isfile(d.weights_path):
            model.load_state_dict(
                torch.load(d.weights_path, weights_only=True, map_location=device)
            )
            epsilon = 0.2
        else:
            epsilon = 1.0
            for p in model.parameters():
                nn.init.normal_(p)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

    for ep in range(d.num_episodes):
        log.memories.clear()
        model.eval()
        run_episode(log, model, epsilon, collect=True)
        model.train()
        losses = []
        for e in range(d.epochs):
            batch = model.tensorify(log.memories, d.batch_size)
            loss = run_epoch(model, batch, optimizer, loss_fn, gamma)
            losses.append(loss)
        print("training in episode", ep, "loss =", np.mean(loss))
        epsilon = max(0.2, epsilon * 0.75)

    log.memories.clear()
    model.eval()
    run_episode(log, model, 0.01, collect=False)
    torch.save(model.state_dict(), "./weights/model_end.pt")


if __name__ == "__main__":
    main()
