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
from dataclasses import dataclass

#
from regi_py.core import PhaseInfo
from regi_py import GameState, DummyLog, CXXConsoleLog
from regi_py import get_strategy_map
from regi_py.strats import RandomStrategy
from regi_py.rl import BatchedMCTS, MCTS, MC2Model, MC1Model, MCTSTesterStrategy


def MCTSLoss(lprob, v, lprob_hat, v_hat):
    n = v.shape[0]
    # loss1 = nn.functional.mse_loss(v, v_hat)
    loss1 = nn.functional.mse_loss(v, v_hat)
    # loss2 = nn.functional.mse_loss(prob, prob_hat)
    loss2 = nn.functional.kl_div(
        input=lprob_hat, target=lprob, reduction="batchmean", log_target=True
    )
    # loss2 = torch.sum(lprob.exp() * lprob_hat) / n
    return loss1, loss2


def run_epoch(model, batch, optimizer):
    states, lprob, v = batch
    lprob_hat, v_hat = model(states)
    loss1, loss2 = MCTSLoss(lprob, v, lprob_hat, v_hat)
    loss = loss1 + loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss1.item(), loss2.item()


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
    print("episode", episode, "saved model", file=sys.stderr)


@dataclass(slots=True)
class PresetGame:
    phase: PhaseInfo
    best: int

def make_preset_games(num_games):
    preset_games = []
    log = DummyLog()
    for _ in range(num_games):
        game = GameState(log)
        num_players = random.randint(2, 4)
        for i in range(num_players):
            game.add_player(RandomStrategy())
        game.initialize()
        preset_games.append(PresetGame(game.export_phaseinfo(), 0))
    return preset_games


def improved_gameplay(
    episode, new_model, old_model, num_simulations, threshold=0.6, preset_games=None
):
    new_model.eval()
    old_model.eval()
    log1 = EndGameLog()
    log2 = EndGameLog()

    newer_better = 0

    for s in range(num_simulations):
        game1 = GameState(log1)
        game2 = GameState(log2)
        #
        if preset_games is None:
            num_players = random.randint(2, 4)
            for i in range(num_players):
                game1.add_player(MCTSTesterStrategy(old_model))
                game2.add_player(MCTSTesterStrategy(new_model))
            game1.initialize()
            game2._init_phaseinfo(game1.export_phaseinfo())
        else:
            num_players = preset_games[s].phase.num_players
            for i in range(num_players):
                game1.add_player(MCTSTesterStrategy(old_model))
                game2.add_player(MCTSTesterStrategy(new_model))
            game1._init_phaseinfo(preset_games[s].phase)
            game2._init_phaseinfo(preset_games[s].phase)
        #
        game1.start_loop()
        game2.start_loop()
        #
        diff1 = log1.e0 - log1.e1
        diff2 = log2.e0 - log2.e1
        #
        if preset_games is None:
            if diff2 >= diff1 and diff2 != 0:
                # print(f"old: {diff1}, new: {diff2} => new is better")
                newer_better += 1
        else:
            if diff2 >= preset_games[s].best and diff2 != 0:
                print(f"old: {preset_games[s].best}, new: {diff2} => new is better", file=sys.stderr)
                newer_better += 1
                preset_games[s].best = diff2
            pass

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


def trainer(tid, shared_model, queue, train_device, test_device, params):
    print(f"P{tid} on {train_device} to train")
    torch.set_num_threads(params.num_threads)
    with torch.device(train_device):
        train_model = MC2Model()
        train_model.device = train_device
        train_model.load_state_dict(shared_model.state_dict())
        train_model = train_model.to(train_device)
        train_model.train()
        loss_fn = MCTSLoss
        optimizer = get_split_optimizer(train_model)

    with torch.device(test_device):
        bench_model = MC2Model()
        bench_model.device = test_device
        bench_model.load_state_dict(shared_model.state_dict())
        bench_model = bench_model.to(test_device)
        bench_model.eval()

    preset_games = make_preset_games(32)
    ep = -1
    samples = []
    prev_best = 0
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

        losses1 = []
        losses2 = []
        for e in range(params.epochs):
            batch = train_model.tensorify(samples, params.batch_size)
            loss1, loss2 = run_epoch(train_model, batch, optimizer)
            losses1.append(loss1)
            losses2.append(loss2)

        print(
            "episode",
            ep,
            f"loss1={np.mean(losses1)}, loss2={np.mean(losses2)}",
            file=sys.stderr,
        )
        bench_model.load_state_dict(train_model.state_dict())
        if improved_gameplay(
            ep,
            new_model=bench_model,
            old_model=shared_model,
            num_simulations=32,
            threshold=0.05,
            preset_games=preset_games,
        ):
            print("episode", ep, "updated model", file=sys.stderr)
            shared_model.load_state_dict(train_model.state_dict())
            if (ep - prev_best) >= params.test_every:
                test_model(ep, shared_model, params.num_simulations)
                prev_best = ep

    torch.save(shared_model.state_dict(), "./weights/model_end.pt")


def explorer(tid, shared_model, queue, device, params):
    print(f"P{tid} on {device} to explore")
    torch.set_num_threads(params.num_threads)
    small_N = params.memory_size // (params.num_processes - 1)
    mcts = MCTS(
        net=shared_model,
        N=small_N,
        batch_size=params.batch_size,
        randomize=False,
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
                f"P{tid},q={queue.qsize()},t={len(examples)},e={examples[-1].value:.4f},dmg={diffe}",
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
        shared_model = MC2Model()
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
