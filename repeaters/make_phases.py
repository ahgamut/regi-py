"""Generate random game starting positions and save them to a file.

The saved positions are reused as a fixed benchmark set by ``run_benchmark.py``, so the
same starting deals can be replayed across many strategy teams / runs.

Each position is a fresh ``GameState.initialize()`` (12 enemies, full deal) for the
chosen player count, exported as its compact phase string (``export_string()``).  The
output JSON records ``num_players`` so the benchmark runner can validate team sizes.
"""

import argparse
import json

from regi_py import DummyLog, GameState, seed as core_seed
from regi_py.strats import RandomStrategy
import random


def make_phases(num_games, num_players, rng_seed=None):
    if rng_seed is not None:
        core_seed(rng_seed)
        random.seed(rng_seed)

    phases = []
    for _ in range(num_games):
        game = GameState(DummyLog())
        for _ in range(num_players):
            game.add_player(RandomStrategy())
        game.initialize()
        phases.append(game.export_string())
    return phases


def main():
    parser = argparse.ArgumentParser(
        "make-phases", description="save random game starting positions for benchmarking"
    )
    parser.add_argument(
        "-n", "--num-games", default=5, type=int, help="number of starting positions"
    )
    parser.add_argument(
        "-p",
        "--num-players",
        default=2,
        type=int,
        choices=(2, 3, 4),
        help="players per game (fixes the deal size)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="target JSON file to write"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="RNG seed for reproducible deals"
    )
    d = parser.parse_args()

    phases = make_phases(d.num_games, d.num_players, d.seed)
    with open(d.output, "w") as f:
        json.dump({"num_players": d.num_players, "phases": phases}, f, indent=2)
    print(f"wrote {len(phases)} phases ({d.num_players}p) to {d.output}")


if __name__ == "__main__":
    main()
