import argparse
import json
import os
import random
import itertools
import time
import multiprocessing as mp
import sys

#
from regi_py import JSONLog, DummyLog, GameState
from regi_py import get_strategy_map
from regi_py.strats import RandomStrategy
from regi_py.strats import BruteSamplingStrategy
from regi_py.strats.mcts_explorer import MCTSExplorerStrategy

STRATEGY_MAP = get_strategy_map()
STRATEGY_MAP.pop("brute")
#
STRATEGY_MAP["mcts-128"] = lambda: MCTSExplorerStrategy(iterations=128)
STRATEGY_MAP["mcts-256"] = lambda: MCTSExplorerStrategy(iterations=256)
STRATEGY_MAP["mcts-16"] = lambda: MCTSExplorerStrategy(iterations=16)
STRATEGY_MAP["mcts-32"] = lambda: MCTSExplorerStrategy(iterations=32)
STRATEGY_MAP["mcts-64"] = lambda: MCTSExplorerStrategy(iterations=64)
STRATEGY_MAP["brute-128"] = lambda: BruteSamplingStrategy(iterations=128)
STRATEGY_MAP["brute-256"] = lambda: BruteSamplingStrategy(iterations=256)
STRATEGY_MAP["brute-16"] = lambda: BruteSamplingStrategy(iterations=16)
STRATEGY_MAP["brute-32"] = lambda: BruteSamplingStrategy(iterations=32)
STRATEGY_MAP["brute-64"] = lambda: BruteSamplingStrategy(iterations=64)


def create_teams(num_teams, num_players):
    bots = list(STRATEGY_MAP.keys())
    default_bots = (
        "random",
        "damage",
        "preserve",
        "mcts-16",
        "mcts-32",
        "mcts-64",
        "mcts-128",
        "mcts-256",
        "brute-16",
        "brute-32",
        "brute-64",
        "brute-128",
        "brute-256",
    )
    default_teams = [tuple([x] * num_players) for x in default_bots]

    teams = set(default_teams)
    while len(teams) < (num_teams + len(default_teams)):
        team = tuple(random.choices(bots, k=num_players))
        avoid = False
        for x0 in teams:
            if len(set(x0) & set(team)) >= 2:
                avoid = True
                break
        if avoid:
            continue
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
    s0 = sum(max(e.hp, 0) for e in game.enemy_pile)
    game.start_loop()
    dt = time.time() - a
    s1 = sum(max(e.hp, 0) for e in game.enemy_pile)
    progress = s0 - s1
    t1 = "|".join(team)
    print(f"{i},{t1},{game.phase_count}p,{dt:.4f}s,{progress}", file=sys.stderr)


def run_game_from_q(tid, output_folder, queue, fill_event):
    fill_event.wait()
    print(tid, "started running games in queue", file=sys.stderr)
    while fill_event.is_set():
        while not queue.empty():
            data = queue.get()
            start_phase, team, i, j, k = data
            try:
                save_single_game(output_folder, start_phase, team, i, j, k)
            except Exception as e:
                print(f"failed  {i, j, k} due to", e)


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


def should_postpone(team):
    for strat in team:
        if "mcts-" in strat or "brute" in strat:
            return True
    return False


def run_simulations(tid, mappings, output_folder, queue, fill_event):
    ral = lambda x: range(len(x))
    start_phases = mappings["games"]
    teams = mappings["teams"]
    num_simulations = mappings["num_simulations"]
    for i, j in itertools.product(ral(start_phases), ral(teams)):
        thr = should_postpone(teams[j])
        for k in range(num_simulations):
            if thr:
                data = (start_phases[i], teams[j], i + 1, j + 1, k + 1)
                queue.put(data)
                fill_event.set()
            else:
                save_single_game(
                    output_folder, start_phases[i], teams[j], i + 1, j + 1, k + 1
                )

    fill_event.clear()


def submain(mappings, output_folder, num_processes, queue_size):
    mp.set_start_method("fork", force=True)
    queue = mp.Queue(maxsize=queue_size)
    fill_event = mp.Event()
    processes = []

    prod = mp.Process(
        target=run_simulations,
        args=(0, mappings, output_folder, queue, fill_event),
    )
    processes.append(prod)
    #
    for j in range(num_processes):
        eater = mp.Process(
            target=run_game_from_q,
            args=(j, output_folder, queue, fill_event),
        )
        eater.start()
        processes.append(eater)

    prod.start()
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
