"""Benchmark strategy *teams* over a fixed set of starting positions.

Runs every ``(starting position) x (team) x (repetition)`` game and collects the game
JSON logs into a single ZIP.  Uses a controlled seed so that, for a given repetition,
**every team plays under the same seed** -- teams are compared on equal footing.

Inputs:
* ``--phases``  : a file from ``make_phases.py`` (``{"num_players", "phases"}``).
* ``--teams``   : a JSON file ``{"teams": [[entry, ...], ...]}`` where each ``entry`` is
                  a strategy spec understood by ``regi_py.build_strategy`` -- a bare name
                  (``"random"``), a ``"NAME-ITERS"`` search spec (``"brute-128"``,
                  ``"mcts-64"``), or an NN dict ``{"name", "iters", "weights"}``.
                  Every team must have exactly ``num_players`` entries.
* ``--reps``    : repetitions per (position, team).
* ``--output``  : target ``.zip``.

Multiprocessing: one *delegator* enumerates the work, N *workers* run the games (each
serializing its own log to JSON so only strings cross the queue), and one *saver* writes
the JSON into the ZIP.  torch is imported and pinned to CPU / a single thread only when
the team file actually uses an NN strategy.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import zipfile

from regi_py import GameState, RegiEncoder, seed as core_seed
from regi_py import build_strategy, spec_uses_nn
from regi_py.strat_spec import parse_spec
from regi_py.logging import JSONBaseLog

# task-queue / result-queue sentinel: "no more items"
DONE = None


class ListLog(JSONBaseLog):
    """In-memory log: collects game events into a list (serialized by the worker).

    ``JSONBaseLog`` already implements every event method in terms of ``log()``; we
    just accumulate instead of writing to a file.
    """

    def __init__(self):
        super().__init__()
        self.events = []

    def log(self, obj):
        self.events.append(obj)


def _team_key(spec):
    return json.dumps(spec, sort_keys=True)


def build_team(team, nn_cache):
    """Instantiate one strategy per seat.  NN builds (expensive weight loads) are
    cached per worker and reused across seats/games; torch-free strats are cheap, so
    they are built fresh."""
    strats = []
    for spec in team:
        if spec_uses_nn(spec):
            key = _team_key(spec)
            strat = nn_cache.get(key)
            if strat is None:
                strat = build_strategy(spec)
                nn_cache[key] = strat
        else:
            strat = build_strategy(spec)
        strats.append(strat)
    return strats


def run_one_game(phase_string, team, seed, nn_cache):
    """Run a single game from ``phase_string`` with ``team`` under ``seed``.

    Returns the JSON string of the game's event log.
    """
    log = ListLog()
    game = GameState(log)
    for strat in build_team(team, nn_cache):
        game.add_player(strat)
    game.init_string(phase_string)
    # Seed BOTH RNGs right before play: MCTS/AZ pull the C++ rng AND Python ``random``.
    core_seed(seed)
    random.seed(seed)
    game.start_loop()
    phase_count = game.phase_count
    enemy_hp_left = sum(max(e.hp, 0) for e in game.enemy_pile)
    # Serialize here, while the live C++ objects referenced by the events are in scope.
    payload = json.dumps(log.events, cls=RegiEncoder)
    return payload, phase_count, enemy_hp_left


def worker(wid, task_queue, result_queue, uses_nn):
    if uses_nn:
        import torch  # lazy: only when an NN strategy is in play

        torch.set_num_threads(1)
    nn_cache = {}
    while True:
        item = task_queue.get()
        if item is DONE:
            break
        name, phase_string, team, seed = item
        try:
            a = time.time()
            payload, phase_count, enemy_hp_left = run_one_game(
                phase_string, team, seed, nn_cache
            )
            result_queue.put((name, payload))
            print(
                f"{name} ok {phase_count}p hp_left={enemy_hp_left} {time.time() - a:.3f}s",
                file=sys.stderr,
            )
        except Exception as e:  # one bad game must not stall the whole run
            print(f"{name} FAILED: {e!r}", file=sys.stderr)


def delegator(task_queue, phases, teams, seeds, num_workers):
    for i, phase_string in enumerate(phases):
        for j, team in enumerate(teams):
            for k, seed in enumerate(seeds):
                name = f"game{i:04d}-team{j:03d}-rep{k:03d}.json"
                task_queue.put((name, phase_string, team, seed))
    for _ in range(num_workers):
        task_queue.put(DONE)


def saver(output, manifest, result_queue):
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        n = 0
        while True:
            item = result_queue.get()
            if item is DONE:
                break
            name, payload = item
            zf.writestr(name, payload)
            n += 1
    print(f"saved {n} games to {output}", file=sys.stderr)


def load_inputs(phases_path, teams_path):
    with open(phases_path) as f:
        pdata = json.load(f)
    num_players = pdata["num_players"]
    phases = pdata["phases"]

    with open(teams_path) as f:
        teams = json.load(f)["teams"]

    for j, team in enumerate(teams):
        if len(team) != num_players:
            raise ValueError(
                f"team {j} has {len(team)} seats but phases are {num_players}-player: {team!r}"
            )
    return num_players, phases, teams


def validate_nn_weights(teams):
    """Fail fast (torch-free) on NN specs whose weights are missing/absent.

    Building an NN strategy needs a weights file, but that build happens lazily in a
    worker; without this check a bad path would fail on every NN game after the whole
    process fleet is already up.  This stays name-based -- it never imports torch."""
    for j, team in enumerate(teams):
        for spec in team:
            if not spec_uses_nn(spec):
                continue
            name, _iters, weights = parse_spec(spec)
            if not weights:
                raise ValueError(
                    f"team {j}: NN strategy {name!r} needs a weights path "
                    f'(use a dict entry {{"name": "{name}", "iters": N, "weights": PATH}})'
                )
            if not os.path.isfile(weights):
                raise ValueError(
                    f"team {j}: weights file for {name!r} not found: {weights!r}"
                )


def make_seeds(reps, base_seed):
    if base_seed is not None:
        return [base_seed + r for r in range(reps)]
    return [random.randrange(2**31) for _ in range(reps)]


def main():
    parser = argparse.ArgumentParser(
        "run-benchmark",
        description="run strategy teams over saved starting positions into a ZIP of game JSON",
    )
    parser.add_argument(
        "-p", "--phases", required=True, help="phases file from make_phases.py"
    )
    parser.add_argument(
        "-t", "--teams", required=True, help="JSON file of teams to test"
    )
    parser.add_argument(
        "-r", "--reps", default=3, type=int, help="repetitions per (position, team)"
    )
    parser.add_argument("-o", "--output", required=True, help="target .zip file")
    parser.add_argument(
        "--num-workers",
        default=max(2, (os.cpu_count() or 2) - 1),
        type=int,
        help="number of worker processes",
    )
    parser.add_argument(
        "--base-seed",
        default=None,
        type=int,
        help="base seed for reproducible repetitions (rep r uses base+r)",
    )
    d = parser.parse_args()

    num_players, phases, teams = load_inputs(d.phases, d.teams)
    validate_nn_weights(teams)
    seeds = make_seeds(d.reps, d.base_seed)
    uses_nn = any(spec_uses_nn(spec) for team in teams for spec in team)

    manifest = {
        "num_players": num_players,
        "num_phases": len(phases),
        "teams": teams,
        "reps": d.reps,
        "seeds": seeds,
        "uses_nn": uses_nn,
        "naming": "game{phase:04d}-team{team:03d}-rep{rep:03d}.json",
    }

    mp.set_start_method("fork", force=True)
    task_queue = mp.Queue(maxsize=4 * d.num_workers)
    result_queue = mp.Queue(maxsize=4 * d.num_workers)

    save_proc = mp.Process(target=saver, args=(d.output, manifest, result_queue))
    save_proc.start()

    workers = [
        mp.Process(target=worker, args=(w, task_queue, result_queue, uses_nn))
        for w in range(d.num_workers)
    ]
    for w in workers:
        w.start()

    deleg = mp.Process(
        target=delegator, args=(task_queue, phases, teams, seeds, d.num_workers)
    )
    deleg.start()

    deleg.join()
    for w in workers:
        w.join()
    # all games produced -> tell the saver to finish.
    result_queue.put(DONE)
    save_proc.join()

    total = len(phases) * len(teams) * d.reps
    print(f"done: {total} games ({len(phases)} phases x {len(teams)} teams x {d.reps} reps)")


if __name__ == "__main__":
    main()
