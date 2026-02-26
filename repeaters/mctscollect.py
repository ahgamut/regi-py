import argparse
import json
import os
import random
import itertools
import time
import multiprocessing as mp
import dataclasses
import msgpack

#
from regi_py import JSONLog, DummyLog, GameState
from regi_py.strats.mcts_explorer import MCTSSaverStrategy


def run_single_game(tid, i, num_bots, num_iterations):
    a = time.time()
    log = DummyLog()
    strat = MCTSSaverStrategy(iterations=num_iterations)
    game = GameState(log)
    for _ in range(num_bots):
        game.add_player(strat)
    game.initialize()
    s0 = sum(max(x.hp, 0) for x in game.enemy_pile)
    game.start_loop()
    s1 = sum(max(x.hp, 0) for x in game.enemy_pile)
    dt = time.time() - a
    end_phase = game.export_phaseinfo()
    history = strat.history
    win = float(end_phase.game_endvalue == 1)
    diff = (360 - s1) / 360
    for info in history:
        info.value = diff
    print(f"{tid, i},{game.phase_count}p,{s0,s1},{dt:.4f}s,{win}")
    return history


def run_mcts_game(tid, num_games, num_bots, num_iterations, output_folder):
    fname = os.path.join(output_folder, f"mcts-{tid}.bin")
    bfile = open(fname, "wb")
    packer = msgpack.Packer()
    for i in range(num_games):
        history = run_single_game(tid, i, num_bots, num_iterations)
        for x in history:
            bfile.write(packer.pack(dataclasses.asdict(x)))
    bfile.close()


def submain(num_games, num_bots, num_iterations, num_processes, output_folder):
    mp.set_start_method("spawn", force=True)
    processes = []

    #
    for j in range(num_processes):
        proc = mp.Process(
            target=run_mcts_game,
            args=(j, num_games, num_bots, num_iterations, output_folder),
        )
        proc.start()
        processes.append(proc)

    for p in processes:
        p.join()


def main():
    parser = argparse.ArgumentParser("sim-repeater")
    parser.add_argument(
        "-n", "--num-games", default=5, type=int, help="number of games"
    )
    parser.add_argument(
        "-b", "--num-bots", default=2, type=int, help="number of bots per team"
    )
    parser.add_argument(
        "-s",
        "--num-iterations",
        default=128,
        type=int,
        help="number of MCTS iterations per phase",
    )
    parser.add_argument(
        "-o", "--output-folder", type=str, required=True, help="folder to store outputs"
    )
    parser.add_argument("--num-processes", default=2, type=int, help="num processes")
    d = parser.parse_args()
    if d.num_bots < 2 or d.num_bots > 4:
        raise RuntimeError(f"can only have 2-4 bots per team, not {d.num_bots}")
    #
    if os.path.exists(d.output_folder):
        if os.path.isfile(d.output_folder):
            raise RuntimeError(f"{output_folder} is a file!")
        os.makedirs(d.output_folder, exist_ok=True)
    else:
        os.makedirs(d.output_folder, exist_ok=False)
    #
    submain(d.num_games, d.num_bots, d.num_iterations, d.num_processes, d.output_folder)


if __name__ == "__main__":
    main()
