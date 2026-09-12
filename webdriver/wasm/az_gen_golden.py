"""Generate a golden fixture for the JS AZ Direct-net bot (torch env).

The AZ analogue of gen_golden.py. For many real decisions it records the serialized
DECISION phase and the combo index the Python AZ NetDirectStrategy would play:
  attack  argmax over a_hat[cell_of_bitwise(combo.bitwise)]  (the (56,22) grid)
  defense argmax over max(0, 1 - sum(k_hat[card.location]))  (keepy fallback)
computed WITHOUT perspectivize (history = [phase]*max_history), so the browser's
AZBot.buildFeeds(..., {reshuffle:false}) featurizes the byte-identical phase and
check_golden.mjs (--paradigm az) asserts the same index -- featurizer + ONNX + argmax
parity end to end.

  python -m webdriver.wasm.az_gen_golden --net basic --weights weights/best_basic.pt \
      --out webdriver/wasm/golden/basic.json --n 300

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
from regi_py.combomap import cell_of_bitwise
from regi_py.strats.phase_utils import PhaseExpander
from regi_py.rl.az.nets import get_net, net_names


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
    """NetDirectStrategy's reshuffle:false path: history = [phase]*max_history, one
    predict, argmax over the offered combos. Attack reads the (56,22) grid via the
    combomap; defense reads the keepyness head (AZ has no defense policy head)."""
    history = [phase] * net.max_history
    _, k_hat, a_hat = net.predict(history)  # perspective -> phase.active_player
    scores = np.zeros(len(combos), dtype=np.float32)
    if phase.phase_attacking:
        for i, combo in enumerate(combos):
            lp = cell_of_bitwise(combo.bitwise)
            if lp is not None:
                scores[i] = a_hat[lp]
    else:
        for i, combo in enumerate(combos):
            wt = sum(float(k_hat[card.location]) for card in combo.parts)
            scores[i] = max(0.0, 1.0 - wt)
    return int(np.argmax(scores)), len(combos)


def generate(net_name, weights_path, out_path, n, seed):
    if net_name not in net_names():
        raise SystemExit(f"{net_name!r} is not an AZ net; choices: {net_names()}")
    net = get_net(net_name)()
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.eval()

    cases = []
    for num_players in (2, 3, 4):
        for s in _collect_phase_strings(seed + num_players, n_games=25, num_players=num_players):
            if len(cases) >= n:
                break
            expander = PhaseExpander(PhaseInfo.from_string(s))
            combos = expander.offered()
            if not combos:
                continue
            # Featurize the phase AT THE DECISION (a loaded phase can auto-resolve and
            # advance to a decision on a different seat) -- same rationale as gen_golden.py.
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
        json.dump({"net": net_name, "paradigm": "az", "seed": seed,
                   "max_history": net.max_history, "cases": cases}, fh)
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
