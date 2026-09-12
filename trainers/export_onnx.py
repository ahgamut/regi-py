"""Export trained Regicide nets to ONNX for the client-side WASM webapp.

The browser bot runs each net's ``forward()`` under onnxruntime-web: the C++/WASM
featurizer (shared with training via the pybind/Embind kernels) builds the input
tensors, ONNX runs the forward pass, and JS does the ``argmax`` (AZ) or the
``[:K]`` softmax + ``argmax`` (ADZ). So this script exports ``forward`` ONLY -- NOT
``predict`` (its featurization is the WASM kernels' job and the ADZ candidate
softmax is JS). Inputs/outputs use FIXED, web-friendly shapes (batch 1, the padded
``MAX_CANDIDATES`` candidate axis) so the graph carries no data-dependent control
flow.

Paradigms (registries are disjoint, so ``--net`` is unambiguous):
  * AZ card-space nets emit ``(v (1,1), k (1,56), a (1,1,56,22) already-softmaxed)``.
      - conv trunk (``basic``, ``attntrunk``): inputs ``location`` / ``used_pile`` /
        ``capability`` (each ``(1,8,56,W)`` frames-as-channels).  GroupNorm -> opset 18.
      - card-token trunk (``percardmlp``, ``cardtx``, ``mixer``): input ``tokens (1,56,264)``.
  * ADZ candidate-scoring nets emit ``(value (1,1), cand_logits (1,128) raw with -inf
    on padded/masked slots, keepy (1,56))``.  Input ``tokens (1,56,264)`` +
    ``cand_feats (1,128,9)`` + ``cand_mask (1,128)`` + a per-net membership encoding
    (``adzmulti``: ``cand_members (1,128,56)``; ``adzpool``: ``cand_idx (1,128,7)`` int64
    + ``cand_partmask (1,128,7)``).

``movetoken`` is intentionally UNSUPPORTED: its in-place advanced-indexed scatter is
export-hostile and its checkpoint is a training FAIL.

Usage (torch env; regi_py installed):
  python -m trainers.export_onnx export --net adzpool --weights weights/best_adzpool.pt \
      --out dist/adzpool.onnx --verify
  python -m trainers.export_onnx export-all --weights-dir weights --out-dir dist --verify

``--verify`` needs ``onnxruntime`` (Python) and cross-checks the ONNX graph against
torch over many real phases: forward outputs within tolerance AND the Direct-net
chosen combo index identical.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from regi_py.core import GameState, RandomStrategy, PhaseInfo
from regi_py.core import seed as core_seed
from regi_py.logging import DummyLog
from regi_py.strats.phase_utils import PhaseExpander
from regi_py.combomap import cell_of_bitwise
from regi_py.rl.az.nets import get_net, net_names
from regi_py.rl.az.nets.base import BaseNet
from regi_py.rl.adz.nets import get_adz_net, adz_net_names
from regi_py.rl.adz.nets.base import CandidateBaseNet

# conv trunk (GroupNorm) needs opset >= 18; TransformerEncoder (SDPA) needs >= 17.
# 18 covers both, so every net exports at one opset.
OPSET = 18
# movetoken: export-hostile in-place scatter + FAIL checkpoint (see module docstring).
UNSUPPORTED = frozenset({"movetoken"})


# ----------------------------------------------------------------------------
# net construction
# ----------------------------------------------------------------------------
def build_net(name):
    """Instantiate the net class for ``name`` (AZ or ADZ), un-built (no weights)."""
    if name in UNSUPPORTED:
        raise SystemExit(f"{name!r} is not ONNX-exportable (see export_onnx docstring)")
    if name in adz_net_names():
        return get_adz_net(name)(), "adz"
    if name in net_names():
        return get_net(name)(), "az"
    choices = ", ".join(n for n in (net_names() + adz_net_names()) if n not in UNSUPPORTED)
    raise SystemExit(f"unknown --net {name!r}; choices: {choices}")


def load_net(name, weights_path):
    net, paradigm = build_net(name)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    # TransformerEncoder's nested-tensor fast path is export-hostile; the plain
    # path (eval + no src_key_padding_mask + this flag) exports cleanly. dropout is
    # already 0.0 in every net, so train/eval forward is identical regardless.
    # ``use_nested_tensor`` is the cached decision the forward actually reads
    # (derived from ``enable_nested_tensor`` at construction), so flip BOTH.
    for mod in net.modules():
        if isinstance(mod, nn.TransformerEncoder):
            mod.enable_nested_tensor = False
            if hasattr(mod, "use_nested_tensor"):
                mod.use_nested_tensor = False
    return net, paradigm


class _Wrapper(nn.Module):
    """Adapt ``net.forward(dict)`` to positional tensors so ONNX gets clean, named
    inputs (and no dict-as-kwargs ambiguity). ``input_keys`` fixes the arg order,
    shared by the exporter, the sidecar contract, and the verifier."""

    def __init__(self, net, input_keys):
        super().__init__()
        self.net = net
        self.input_keys = list(input_keys)

    def forward(self, *args):
        return self.net({k: a for k, a in zip(self.input_keys, args)})


# ----------------------------------------------------------------------------
# real-phase sample generation (correct dtypes/shapes for tracing + verify)
# ----------------------------------------------------------------------------
def _collect_decisions(seeded, n_games=40, num_players=2):
    """Serialized phases from real random self-play, split by attack/defense so the
    verifier exercises both. Mirrors tests/test_features_parity._collect_phases."""
    core_seed(seeded)
    attack, defense = [], []
    for _ in range(n_games):
        game = GameState(DummyLog())
        for _ in range(num_players):
            game.add_player(RandomStrategy())
        game.initialize()
        game.start_loop()
        for ph in game.history:
            (attack if ph.phase_attacking else defense).append(ph.to_string())
    return attack, defense


def _phase_input(net, paradigm, phase):
    """Build one net input dict for a serialized ``phase`` via the net's OWN
    tensorify (so dtypes/shapes/normalization match training exactly). Returns
    ``(data, combos)`` where ``combos`` is the offered-combo list (ADZ) or ``[]``.
    ``None`` if the phase offers nothing an ADZ net can score."""
    # raw_window_arrays indexes history[-window..-1] with NO internal padding, so
    # the caller must hand it a window-length history. For a bare phase that is
    # [phase]*max_history -- exactly what trimmed_history([], phase, window) yields
    # (left-pad with the oldest frame), which is what the Direct strategies feed.
    history = [phase] * net.max_history
    if paradigm == "adz":
        combos = PhaseExpander(phase).offered()
        if not combos:
            return None
        data = type(net).tensorify_predict(
            history, combos, phase, phase.active_player, net.max_history
        )
        return data, combos
    # az: no candidate list; the grid head is featurized from the window alone
    data = type(net).tensorify_phases(history, phase.active_player, net.max_history)
    return data, []


def _first_sample(net, paradigm):
    """One representative input dict for tracing the export (any real decision)."""
    attack, _ = _collect_decisions(1234)
    for s in attack:
        phase = PhaseInfo.from_string(s)
        got = _phase_input(net, paradigm, phase)
        if got is not None:
            return got[0]
    raise SystemExit("could not build a sample input (no offered combos found)")


# ----------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------
def _inline_external_data(out_path):
    """Some torch exporters write weights to a sibling ``<model>.onnx.data`` (or
    ``.data``) external-data file. For static web hosting a single self-contained
    ``.onnx`` is simpler (one fetch, no external-data registration in ORT-web), and
    these nets are all <1 MB, so fold the weights back in and drop the sidecar."""
    import onnx  # available in any env that ran torch.onnx.export

    model = onnx.load(out_path)  # load_external_data=True by default -> weights in-mem
    onnx.save_model(model, out_path, save_as_external_data=False)
    for cand in (out_path + ".data", os.path.splitext(out_path)[0] + ".data"):
        if os.path.exists(cand):
            os.remove(cand)


def export(name, weights_path, out_path, verify=False, external_data=False):
    net, paradigm = load_net(name, weights_path)
    sample = _first_sample(net, paradigm)
    input_keys = list(sample.keys())
    out_names = ["value", "cand_logits", "keepy"] if paradigm == "adz" else ["v", "k", "a"]
    wrapper = _Wrapper(net, input_keys)
    wrapper.eval()

    args = tuple(sample[k] for k in input_keys)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            args,
            out_path,
            input_names=input_keys,
            output_names=out_names,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,  # FIXED shapes -> a static, web-friendly graph
        )
    if not external_data:
        _inline_external_data(out_path)

    contract = {
        "net": name,
        "paradigm": paradigm,
        "opset": OPSET,
        "inputs": [
            {"name": k, "shape": list(sample[k].shape), "dtype": str(sample[k].dtype).replace("torch.", "")}
            for k in input_keys
        ],
        "outputs": out_names,
        "max_history": net.max_history,
    }
    contract_path = os.path.splitext(out_path)[0] + ".io.json"
    with open(contract_path, "w") as fh:
        json.dump(contract, fh, indent=2)

    print(f"exported {name} -> {out_path}  ({os.path.getsize(out_path)} bytes)")
    print(f"  io contract -> {contract_path}")
    if verify:
        verify_onnx(name, weights_path, out_path)


# ----------------------------------------------------------------------------
# verification: ONNX graph == torch forward, and same Direct-net chosen index
# ----------------------------------------------------------------------------
def _direct_index_adz(cand_logits, K):
    """ADZ Direct-net choice: softmax over the K real candidates, argmax (== argmax
    of the raw logits, but computed as ADZDirectStrategy does)."""
    logits = cand_logits[0, :K]
    priors = np.exp(logits - logits.max())
    priors /= priors.sum()
    return int(np.argmax(priors))


def _direct_index_az_attack(a_hat, combos):
    """AZ attack Direct-net choice: read each offered combo's (loc,pst) cell prior
    from the already-softmaxed (56,22) grid, argmax (NetDirectStrategy.getAttackIndex)."""
    grid = a_hat[0, 0]  # (56, 22)
    scores = np.zeros(len(combos), dtype=np.float32)
    for i, combo in enumerate(combos):
        lp = cell_of_bitwise(combo.bitwise)
        if lp is not None:
            scores[i] = grid[lp]
    return int(np.argmax(scores))


def verify_onnx(name, weights_path, onnx_path, atol=1e-4, rtol=1e-4):
    import onnxruntime as ort  # local: only the verify path needs it

    net, paradigm = load_net(name, weights_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_order = [o.name for o in sess.get_outputs()]

    attack, defense = _collect_decisions(777)
    corpus = attack[:250] + defense[:250]
    checked = 0
    per_out_diff = {}  # output name -> running max |torch - ort| over finite cells
    idx_mismatch = 0
    for s in corpus:
        phase = PhaseInfo.from_string(s)
        got = _phase_input(net, paradigm, phase)
        if got is None:
            continue
        data, combos = got
        feeds = {k: v.detach().cpu().numpy() for k, v in data.items()}
        ort_out = dict(zip(out_order, sess.run(out_order, feeds)))
        with torch.inference_mode():
            t_out = net(data)
        t_named = dict(zip(
            ["value", "cand_logits", "keepy"] if paradigm == "adz" else ["v", "k", "a"],
            [t.detach().cpu().numpy() for t in t_out],
        ))

        # numeric forward parity, ignoring the -inf padded ADZ logit slots (both
        # sides produce -inf there; the finite real entries are what JS reads)
        for key, tv in t_named.items():
            ov = ort_out[key]
            finite = np.isfinite(tv) & np.isfinite(ov)
            if finite.any():
                diff = np.abs(tv[finite].astype(np.float64) - ov[finite].astype(np.float64)).max()
                per_out_diff[key] = max(per_out_diff.get(key, 0.0), float(diff))
                assert np.allclose(tv[finite], ov[finite], atol=atol, rtol=rtol), (
                    f"{name}: forward mismatch on {key!r}, max|diff|={diff}")
            # non-finite cells must agree (both -inf on the same padded slots)
            assert np.array_equal(np.isfinite(tv), np.isfinite(ov)), (
                f"{name}: -inf mask disagreement on {key!r}")

        # Direct-net chosen-index parity (the decision the webapp bot actually makes)
        if paradigm == "adz":
            K = len(combos)
            t_idx = _direct_index_adz(t_named["cand_logits"], K)
            o_idx = _direct_index_adz(ort_out["cand_logits"], K)
        else:
            if not phase.phase_attacking:
                checked += 1
                continue  # AZ defense uses the keepy fallback; index parity covered by k parity
            combos = PhaseExpander(phase).offered()
            if not combos:
                checked += 1
                continue
            t_idx = _direct_index_az_attack(t_named["a"], combos)
            o_idx = _direct_index_az_attack(ort_out["a"], combos)
        idx_mismatch += int(t_idx != o_idx)
        checked += 1

    assert checked > 100, f"{name}: only {checked} phases verified"
    assert idx_mismatch == 0, f"{name}: {idx_mismatch}/{checked} Direct-net index mismatches"
    # per-output max diff: raw ADZ cand_logits sit ~1e-4 (unbounded logits through an
    # einsum), while squashed heads (tanh/sigmoid/softmax) sit ~1e-6 -- both benign,
    # since the Direct-net argmax is bit-identical across every phase.
    diffs = ", ".join(f"{k}={v:.1e}" for k, v in per_out_diff.items())
    print(f"  verify OK: {checked} phases, index parity exact | max|diff| per output: {diffs}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
_WEIGHT_PATTERNS = ("best_{net}.pt", "best-r2-{net}.pt", "best-r1-{net}.pt")


def _find_weights(weights_dir, net):
    for pat in _WEIGHT_PATTERNS:
        p = os.path.join(weights_dir, pat.format(net=net))
        if os.path.exists(p):
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="export one net to ONNX")
    e.add_argument("--net", required=True)
    e.add_argument("--weights", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--verify", action="store_true")
    e.add_argument("--external-data", action="store_true",
                   help="keep weights in a sibling .onnx.data file (default: inline single-file)")

    a = sub.add_parser("export-all", help="export every supported net found in --weights-dir")
    a.add_argument("--weights-dir", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--verify", action="store_true")
    a.add_argument("--external-data", action="store_true",
                   help="keep weights in a sibling .onnx.data file (default: inline single-file)")
    a.add_argument("--nets", nargs="*", default=None,
                   help="restrict to these net names (default: all supported)")

    v = sub.add_parser("verify", help="verify an already-exported ONNX vs torch")
    v.add_argument("--net", required=True)
    v.add_argument("--weights", required=True)
    v.add_argument("--onnx", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "export":
        export(args.net, args.weights, args.out, verify=args.verify,
               external_data=args.external_data)
    elif args.cmd == "verify":
        verify_onnx(args.net, args.weights, args.onnx)
    else:  # export-all
        supported = [n for n in (net_names() + adz_net_names()) if n not in UNSUPPORTED]
        targets = args.nets or supported
        exported, skipped = [], []
        for net in targets:
            if net not in supported:
                print(f"skip {net!r}: unsupported", file=sys.stderr)
                skipped.append(net)
                continue
            wp = _find_weights(args.weights_dir, net)
            if wp is None:
                print(f"skip {net!r}: no weights in {args.weights_dir}", file=sys.stderr)
                skipped.append(net)
                continue
            export(net, wp, os.path.join(args.out_dir, f"{net}.onnx"), verify=args.verify,
                   external_data=args.external_data)
            exported.append(net)
        print(f"\nexported {len(exported)}: {', '.join(exported)}")
        if skipped:
            print(f"skipped {len(skipped)}: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
