"""Generic trainer plumbing for the AlphaZero pipeline.

Lifted out of ``trainers/az_trainer.py`` so that module keeps only the
multiprocessing orchestration (``submain``/``trainer``/``explorer``/``main``).
Everything here is reusable, single-process, and free of any ``mp`` state:
self-play data generation (``run_single_game``), the optimization step
(``run_epoch``), evaluation (``test_model``/``improved_gameplay``), and small
helpers (``EndGameLog``, ``total_enemy_hp``, ``get_split_optimizer``, ``drain``).
"""
import random
import sys
import time

#
import torch
import numpy as np

from regi_py import GameState, DummyLog
from regi_py.strats import RandomStrategy
from regi_py.rl.az_explorer import (
    NetDirectStrategy,
    AlphaZeroNode,
    simulate_node,
)
from regi_py.rl.basicnet import BasicNet
from regi_py.rl.utils import enemy_hp_left, hp_loss_penalty


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


def drain(q, buf):
    while not q.empty():
        obj = q.get()
        evicted = buf.add(obj)
        del obj  # drop child's dict ref
        if evicted is not None:
            del evicted  # retire evicted shard


def run_epoch(model, batch, optimizer):
    # a batch is a tuple of tensors in ``BasicNet.TRAIN_FIELDS`` order (that is
    # how ``ShardBuffer`` packs each shard's ``TensorDataset``)
    data = {k: v.to(model.device) for k, v in zip(BasicNet.TRAIN_FIELDS, batch)}
    v_hat, k_hat, a_hat = model(data)
    v, k, a = data["value"], data["keepyness"], data["atk_probs"]
    loss = model.calculate_loss((v, k, a), (v_hat, k_hat, a_hat), data["attacking"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


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
    node = AlphaZeroNode(start_phase, net=net, history=[], prior=1.0, trim=False)
    s0 = enemy_hp_left(node.root_phase)
    while node.root_phase.game_endvalue == 0:
        simulate_node(node, num_iterations)
        history.append(node.export())
        child = node.best_child_node
        child.parent = None
        node = child
    history.append(node.export())
    win = node.root_phase.game_endvalue == 1
    s1 = enemy_hp_left(node.root_phase)
    #
    dt = time.time() - a
    reward = 1.0 if win else hp_loss_penalty(s1)
    for info in history:
        info.value = reward
    # print(f"{tid},{i},p{len(history)},{s0},{s1},{dt:.4f}s,{win}", file=sys.stderr)
    return BasicNet.tensorify_training(history)


def test_model(episode, model, num_simulations):
    model.eval()
    log = EndGameLog()
    diffe = []
    for s in range(10):
        game = GameState(log)
        num_players = random.randint(2, 4)
        for i in range(num_players):
            game.add_player(NetDirectStrategy(model))
        game.initialize()
        game.start_loop()
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
        game2.init_phaseinfo(game1.export_phaseinfo())
        #
        game1.start_loop()
        game2.start_loop()
        #
        diff1 = log1.e0 - log1.e1
        diff2 = log2.e0 - log2.e1
        #
        print(f"{s} old: {diff1}, new: {diff2}")
        if diff2 > diff1 and diff2 != 0:
            newer_better += 1

    nb_ratio = newer_better / num_simulations
    print(f"{episode} newer better in {100*nb_ratio:.4f}% of games", file=sys.stderr)
    return nb_ratio > threshold
