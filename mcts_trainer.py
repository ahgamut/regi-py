import argparse
import os
import random
import sys

#
import torch
import torch.nn as nn
import numpy as np

#
from regi_py import GameState, DummyLog, CXXConsoleLog
from regi_py import get_strategy_map
from regi_py.rl import MCTS, MC1Model, MCTSTesterStrategy


def MCTSLoss(prob, v, prob_hat, v_hat):
    n = v.shape[0]
    loss1 = nn.functional.mse_loss(v, v_hat)
    # loss2 = nn.functional.mse_loss(prob, prob_hat)
    p1 = prob
    p2 = -nn.functional.log_softmax(prob_hat, dim=1)
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


def test_model(episode, model, num_players, num_simulations):
    model.eval()
    log = EndGameLog()
    game = GameState(log)
    for i in range(num_players):
        game.add_player(MCTSTesterStrategy(model))
    diffe = []
    for s in range(10):
        game._init_random()
        e0 = total_enemy_hp(game)
        game.start_loop()
        e1 = total_enemy_hp(game)
        diffe.append(log.diffe())
    print("test games:", diffe)
    torch.save(model.state_dict(), "./weights/best_model.pt")


def main():
    parser = argparse.ArgumentParser("regi-mcts-trainer")
    parser.add_argument(
        "--num-episodes", default=1, type=int, help="number of episodes"
    )
    parser.add_argument(
        "--num-simulations", default=32, type=int, help="number of simulations per game"
    )
    parser.add_argument("--test-every", default=1, type=int, help="test every k epochs")
    parser.add_argument("--num-players", default=2, type=int, help="number of players")
    parser.add_argument("--memory-size", default=64, type=int, help="memory size")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=1, type=int, help="epochs")
    parser.add_argument(
        "--weights-path", default="./weights/best_model.pt", help="weights"
    )
    d = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with torch.device(device):
        model = MC1Model()
        model.device = device
        loss_fn = MCTSLoss
        optimizer = torch.optim.SGD(model.parameters())

    mcts = MCTS(num_players=d.num_players, net=model, N=d.memory_size)

    for ep in range(d.num_episodes):
        mcts.examples.clear()
        model.eval()
        while not mcts.sim_game_full(sims=d.num_simulations):
            mcts.reset_game()
        model.train()
        losses = []
        for e in range(d.epochs):
            batch = model.tensorify(mcts.examples, d.batch_size)
            loss = run_epoch(model, batch, optimizer)
            losses.append(loss)
        print("training in episode", ep, "loss =", np.mean(loss))
        if ep % d.test_every == 0:
            test_model(ep, model, d.num_players, d.num_simulations)

    torch.save(model.state_dict(), "./weights/model_end.pt")


if __name__ == "__main__":
    main()
