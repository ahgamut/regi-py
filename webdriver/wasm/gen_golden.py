"""Generate a golden fixture for the JS ADZ Direct-net bot (torch env).

For many real ADZ decisions this records the serialized phase and the combo index
the Python ADZDirectStrategy would play -- computed WITHOUT perspectivize (raw
phase, empty history) so the browser's NetBot.buildFeeds(..., {reshuffle:false})
featurizes the byte-identical phase and the check has no RNG to sync. The companion
webdriver/wasm/check_golden.mjs replays each case through NetBot and asserts the
same index, proving featurizer + ONNX + argmax parity end to end.

  python -m webdriver.wasm.gen_golden --net adzpool --weights weights/best_adzpool.pt \
      --out webdriver/wasm/golden/adzpool.json --n 300

Run from the repo root (so `webdriver` is importable), in the torch env.
"""
import argparse
import json
import os

import numpy as np
import torch

from regi_py.core import GameState, RandomStrategy, PhaseInfo
from regi_py.core import seed as core_seed
from regi_py.logging import DummyLog
from regi_py.strats.phase_utils import PhaseExpander
from regi_py.rl.adz.nets import get_adz_net, adz_net_names


def _collect_phase_strings(seeded, n_games, num_players):
    core_seed(seeded)
    out = []
    for _ in range(n_games):
        game = GameState(DummyLog())
        for _ in range(num_players):
            game.add_player(RandomStrategy())
        game.initialize()
        game.start_loop()
        out.extend(ph.to_string() for ph in game.history)
    return out


def _combo_locations(combo):
    return sorted(int(card.location) for card in combo.parts)


def _direct_index(net, phase, combos):
    """Exactly NetBot's reshuffle:false path: raw phase, history = [phase]*max_history,
    predict, argmax over the offered combos' priors. No perspectivize."""
    history = [phase] * net.max_history
    _, priors = net.predict(history, combos, phase)  # perspective -> phase.active_player
    return int(np.argmax(priors)), len(combos)


def generate(net_name, weights_path, out_path, n, seed):
    if net_name not in adz_net_names():
        raise SystemExit(f"{net_name!r} is not an ADZ net; choices: {adz_net_names()}")
    net = get_adz_net(net_name)()
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.eval()

    cases = []
    # spread across 2/3/4-player games so the window/candidate widths vary
    for num_players in (2, 3, 4):
        for s in _collect_phase_strings(seed + num_players, n_games=25, num_players=num_players):
            if len(cases) >= n:
                break
            expander = PhaseExpander(PhaseInfo.from_string(s))
            combos = expander.offered()
            if not combos:
                continue
            # Featurize the phase AT THE DECISION, not the loaded phase: a loaded
            # phase can auto-resolve (e.g. a full block) and advance to a decision
            # on a different seat before any choice is made, so the loaded phase's
            # perspective would not match the offered combos (this is what the
            # browser's live GameDriver featurizes, and what check_golden.mjs
            # reloads to). Store that decision phase so both sides align.
            phase = expander.decision_phase()
            if phase is None:
                continue
            index, k = _direct_index(net, phase, combos)
            cases.append({
                "phase": phase.to_string(),
                "num_players": int(phase.num_players),
                "attacking": bool(phase.phase_attacking),
                "k": k,
                "index": index,
                "combos": [_combo_locations(c) for c in combos],
            })
        if len(cases) >= n:
            break

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"net": net_name, "seed": seed, "max_history": net.max_history, "cases": cases}, fh)
    atk = sum(c["attacking"] for c in cases)
    print(f"wrote {len(cases)} cases ({atk} attack / {len(cases) - atk} defense) -> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="number of decisions to record")
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args(argv)
    generate(args.net, args.weights, args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
