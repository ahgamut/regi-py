import argparse
import json
import os
import random
import itertools
import time
import multiprocessing as mp

#
from regi_py import JSONLog, DummyLog, GameState
from regi_py import get_strategy_map
from regi_py.strats import RandomStrategy

STRATEGY_MAP = get_strategy_map()


def create_teams(num_teams, num_players):
    bots = list(STRATEGY_MAP.keys())

    teams = set()
    while len(teams) < (num_teams):
        team = tuple(random.choices(bots, k=2))
        teams.add(team)
    return list(teams)


def get_mappings(num_games, num_teams, num_players, num_simulations):
    teams = create_teams(num_teams, num_players)

    start_phases = []
    for i in range(num_games):
        log = DummyLog()
        game = GameState(log)
        for j in range(num_players):
            game.add_player(RandomStrategy())
        game.initialize()
        start_phases.append(game.export_string())

    result = dict()
    result["games"] = start_phases
    result["teams"] = teams
    result["num_simulations"] = num_simulations
    return result


def save_single_game(output_folder, start_phase, team, i, j, k):
    a = time.time()
    fname = f"game{i:04d}-team{j:03d}-sim{k:02d}.json"
    log = JSONLog(os.path.join(output_folder, fname))
    game = GameState(log)
    for bot in team:
        game.add_player(STRATEGY_MAP[bot]())
    game._init_string(start_phase)
    game.start_loop()
    dt = time.time() - a
    print(f"game {i, team} ran for {game.phase_count} phases, {dt:.4f}s")


def run_game_from_q(tid, output_folder, queue):
    while True:
        data = queue.get()
        start_phase, team, i, j, k = data
        try:
            save_single_game(output_folder, start_phase, team, i, j, k)
        except Exception as e:
            print(f"failed  {i, j, k} due to", e)

        if queue.empty() and tid == 0:
            return


def save_config(mappings, output_folder):
    if os.path.exists(output_folder):
        if os.path.isfile(output_folder):
            raise RuntimeError(f"{output_folder} is a file!")
        os.makedirs(output_folder, exist_ok=True)
    else:
        os.makedirs(output_folder, exist_ok=False)

    #
    mappings_filename = os.path.join(output_folder, "mappings.json")
    with open(mappings_filename, "w") as mfile:
        json.dump(mappings, mfile, indent=4)


def run_simulations(tid, mappings, output_folder, queue):
    ral = lambda x: range(len(x))
    start_phases = mappings["games"]
    teams = mappings["teams"]
    num_simulations = mappings["num_simulations"]
    for i, j in itertools.product(ral(start_phases), ral(teams)):
        thr = "mcts-explorer" in teams[j]
        thr = thr or "brute" in teams[j]
        for k in range(num_simulations):
            if thr:
                data = (start_phases[i], teams[j], i + 1, j + 1, k + 1)
                queue.put(data)
            else:
                save_single_game(
                    output_folder, start_phases[i], teams[j], i + 1, j + 1, k + 1
                )
    run_game_from_q(tid, output_folder, queue)


def submain(mappings, output_folder, num_processes, queue_size):
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue(maxsize=queue_size)
    processes = []

    filler = mp.Process(
        target=run_simulations, args=(0, mappings, output_folder, queue)
    )
    filler.start()
    #
    for j in range(1, num_processes):
        eater = mp.Process(target=run_game_from_q, args=(j, output_folder, queue))
        eater.start()
        processes.append(eater)

    filler.join()
    for p in processes:
        p.terminate()


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
        "--num-simulations",
        default=3,
        type=int,
        help="number of simulations per game",
    )
    parser.add_argument(
        "-t",
        "--num-teams",
        default=5,
        type=int,
        help="number of teams to test on each game",
    )
    parser.add_argument(
        "-o", "--output-folder", type=str, required=True, help="folder to store outputs"
    )
    parser.add_argument("-q", "--queue-size", default=0, help="queue size")
    parser.add_argument("--num-processes", default=2, type=int, help="num processes")
    d = parser.parse_args()
    if d.num_bots < 2 or d.num_bots > 4:
        raise RuntimeError(f"can only have 2-4 bots per team, not {d.num_bots}")

    mappings = get_mappings(d.num_games, d.num_teams, d.num_bots, d.num_simulations)
    save_config(mappings, d.output_folder)
    submain(mappings, d.output_folder, d.num_processes, d.queue_size)


if __name__ == "__main__":
    main()
