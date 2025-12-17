import argparse
import os
import random

#
import torch
import torch.nn as nn
import numpy as np

#
from regi_py import GameState
from regi_py import STRATEGY_MAP
from regi_py.rl import MemoryLog, RL1Model

BEST_REMAIN = 350


def bonus_rewards(remaining):
    hps = np.arange(340, -1, 20)
    res = 0
    for i, hp in hps:
        if remaining <= hp:
            res += 1.4**i
    return res


def base_rewards(remaining):
    nr = 360 - remaining
    dr = 360 - BEST_REMAIN
    return 360 * nr / dr


def set_rewards(log, start, end, model):
    last_state = log.memories[end - 1]
    last_state["best_future"] = model.predict(last_state).max().item()
    global BEST_REMAIN
    if last_state["remaining"] <= BEST_REMAIN:
        BEST_REMAIN = last_state["remaining"]
        print("new best!", BEST_REMAIN)
        torch.save(model.state_dict(), "./best_model.pt")
        last_state["reward"] = 500
        last_state["best_from_here"] = 500
        last_state["best_future"] = 0
    else:
        last_state["reward"] = -last_state["remaining"]
        bonus = 0
        if last_state["remaining"] <= 160:
            bonus = 100
        if last_state["remaining"] <= 280:
            bonus = 50
        last_state["best_from_here"] = -150 + bonus

    for i in range((end - 2), (start - 1), -1):
        cur_state = log.memories[i]
        next_state = log.memories[i + 1]
        cur_state["reward"] = base_rewards(cur_state["remaining"]) + bonus_rewards(
            cur_state["remaining"]
        )
        cur_state["best_future"] = next_state["best_from_here"]
        cur_state["best_from_here"] = model.predict(cur_state).max().item()


def basic_game(strats, log, model, epsilon, collect=True):
    n_players = len(strats)
    assert n_players in [2, 3, 4], "only 2, 3, or 4 players"

    print(strats, end=" ")
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
    game.initialize()
    game.start_loop()
    end = len(log.memories)
    set_rewards(log, start, end, model)


def run_epoch(model, batch, optimizer, loss_fn, gamma=0.99):
    q_values = model(batch)
    target_q_values = (batch["rewards"] + gamma * batch["best_future"]) / 36
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
        "--num-episodes", default=1, type=int, help="number of episodes"
    )
    parser.add_argument("--memory-size", default=64, type=int, help="memory size")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=1, type=int, help="epochs")
    parser.add_argument("--weights-path", default="./best_model.pt", help="weights")
    d = parser.parse_args()

    log = MemoryLog(N=d.memory_size)
    model = RL1Model()
    if os.path.isfile(d.weights_path):
        model.load_state_dict(torch.load(d.weights_path, weights_only=True))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    epsilon = 1.0
    gamma = 0.99

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
        epsilon = max(0.2, epsilon * 0.5)

    log.memories.clear()
    model.eval()
    run_episode(log, model, 0.01, collect=False)
    # torch.save(model.state_dict(), "./weights.pt")


if __name__ == "__main__":
    main()
