import argparse
import json
import os
import random
import itertools

#
from regi_py import JSONLog, DummyLog, GameState
from regi_py import get_strategy_map
from regi_py.strats import RandomStrategy

STRATEGY_MAP = get_strategy_map()


def create_teams(num_teams, num_players):
    bots = list(STRATEGY_MAP.keys())

    teams = []
    for _ in range(num_teams):
        team = random.choices(bots, k=2)
        teams.append(team)
    return teams


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
    fname = f"game{i:04d}-team{j:03d}-sim{k:02d}.json"
    log = JSONLog(os.path.join(output_folder, fname))
    game = GameState(log)
    for bot in team:
        game.add_player(STRATEGY_MAP[bot]())
    game._init_string(start_phase)
    game.start_loop()
    print(fname, "ran for", game.phase_count, "phases")


def save_simulations(mappings, output_folder):
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

    ral = lambda x: range(len(x))

    start_phases = mappings["games"]
    teams = mappings["teams"]
    num_simulations = mappings["num_simulations"]
    for i, j in itertools.product(ral(start_phases), ral(teams)):
        for k in range(num_simulations):
            try:
                save_single_game(output_folder, start_phases[i], teams[j], i+1, j+1, k+1)
            except Exception as e:
                print(f"failed  {i, j, k} due to", e)


def main():
    parser = argparse.ArgumentParser("sim-repeater")
    parser.add_argument(
        "-b", "--num-bots", default=2, type=int, help="number of bots per team"
    )
    parser.add_argument(
        "-n", "--num-games", default=5, type=int, help="number of games"
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
    d = parser.parse_args()
    if d.num_bots < 2 or d.num_bots > 4:
        raise RuntimeError(f"can only have 2-4 bots per team, not {d.num_bots}")

    mappings = get_mappings(d.num_games, d.num_teams, d.num_bots, d.num_simulations)
    save_simulations(mappings, d.output_folder)


if __name__ == "__main__":
    main()
