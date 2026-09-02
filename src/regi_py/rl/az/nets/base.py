"""``BaseNet`` -- the contract every AlphaZero architecture satisfies.

The training/self-play/eval pipeline is written against this interface, not any
concrete net, so architectures are swappable via the ``nets`` registry. A
card-space net (input on the 56-card axis, heads emitting value / keepyness(56) /
atk_probs(56,22)) usually only overrides ``_assemble`` (input layout), ``forward``
(trunk + heads), ``TRAIN_FIELDS`` and ``__mname__``; ``predict``,
``tensorify_phases``, ``tensorify_training`` and ``calculate_loss`` are shared
defaults here, driven by the ``_assemble`` hook and ``features.py``.
"""
from regi_py.core import MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS
from regi_py.rl import features

import numpy as np
import torch
import torch.nn as nn


class BaseNet(nn.Module):
    # ---- class contract each concrete net sets ----
    __mname__ = "base"          # checkpoint tag + strat-name suffix; must be unique
    max_history = 8             # history window length
    TRAIN_FIELDS = ()           # input columns THEN target columns, in shard order

    _infer_client = None        # play_server.InferClient, or None for local CPU predict

    def __init__(self):
        super().__init__()
        self.device = "cpu"

    # ---- subclass implements ----
    def forward(self, data):
        """dict of input tensors -> (v, k, a)."""
        raise NotImplementedError

    @classmethod
    def _assemble(cls, loc, usp, cap):
        """Map raw window arrays (each ``(window, 56, W)`` numpy) into this net's
        input-tensor dict, with a leading batch axis of size 1. The dict keys are
        this net's input ``TRAIN_FIELDS``. Overridden per architecture."""
        raise NotImplementedError

    # ---- shared defaults (parameterized by _assemble + features) ----
    @classmethod
    def tensorify_phases(cls, history, perspective=None, window=None):
        if window is None:
            window = cls.max_history
        if perspective is None:
            perspective = history[-1].active_player
        loc, usp, cap = features.raw_window_arrays(history, perspective, window)
        return cls._assemble(loc, usp, cap)

    @classmethod
    def tensorify_training(cls, infos):
        N = len(infos)
        window = len(infos[0].history)
        value = torch.zeros((N, 1))
        keepyness = torch.ones((N, MAX_CARDS_IN_GAME))
        atk_probs = torch.zeros((N, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS))
        attacking = torch.zeros((N, 1))
        input_cols = None
        for i, info in enumerate(infos):
            cur_phase = info.history[-1]
            perspective = cur_phase.active_player
            loc, usp, cap = features.raw_window_arrays(info.history, perspective, window)
            assembled = cls._assemble(loc, usp, cap)  # dict of (1, ...) tensors
            if input_cols is None:
                input_cols = {k: [] for k in assembled}
            for k, v in assembled.items():
                input_cols[k].append(v)
            tgt = features.shared_targets(info, cur_phase)
            value[i, 0] = tgt["value"]
            attacking[i, 0] = tgt["attacking"]
            keepyness[i, :] = torch.from_numpy(tgt["keepyness"])
            atk_probs[i, 0] = torch.from_numpy(tgt["atk_probs"])
        result = {k: torch.cat(v, dim=0) for k, v in input_cols.items()}
        result["value"] = value
        result["keepyness"] = keepyness
        result["atk_probs"] = atk_probs
        result["attacking"] = attacking
        return result

    def predict(self, history, perspective=None):
        if self._infer_client is not None:
            return self.predict_remote(history, perspective)
        # inference_mode: self-play/eval never backprop through predict, so skip
        # all autograd bookkeeping (a real CPU win, since every call otherwise
        # builds a graph that's immediately discarded)
        with torch.inference_mode():
            data = type(self).tensorify_phases(history, perspective, self.max_history)
            v_hat0, k_hat0, a_hat0 = self.forward(data)
            v_hat = float(v_hat0.detach().cpu().numpy()[0, 0])
            k_hat = k_hat0.detach().cpu().numpy()[0, :]
            a_hat = a_hat0.detach().cpu().numpy()[0, 0, :, :]
        return v_hat, k_hat, a_hat

    def predict_remote(self, history, perspective=None):
        data = type(self).tensorify_phases(history, perspective, self.max_history)
        out = self._infer_client.exchange(data)
        v_hat = float(out["v"][0])
        k_hat = out["k"].numpy().astype(np.float32)
        a_hat = out["a"][0].numpy().astype(np.float32)
        return v_hat, k_hat, a_hat

    def predict_batch(self, batch):
        with torch.inference_mode():
            data = {k: v.to(self.device) for k, v in batch.items()}
            v_hat, k_hat, a_hat = self.forward(data)
            return {
                "v": v_hat.detach().to("cpu"),
                "k": k_hat.detach().to("cpu"),
                "a": a_hat.detach().to("cpu"),
            }

    @classmethod
    def sample_predict_input(cls, node):
        return cls.tensorify_phases(node.history, None, cls.max_history)

    def calculate_loss(self, data, y_hat):
        """Shared value + keepyness + masked-policy-CE loss. Architecture-agnostic:
        it only reads the (v, k, a) heads and the attack-phase mask, so every
        card-space net reuses it. Reads its own target keys from ``data`` (the
        batch dict keyed by ``TRAIN_FIELDS``), so ``run_epoch`` stays paradigm-
        agnostic. Returns (total, (policy, value, keepy))."""
        v, k, a = data["value"], data["keepyness"], data["atk_probs"]
        phase_atk = data["attacking"]
        v_hat, k_hat, a_hat = y_hat
        # clamp inside log: masked cells make a_hat exactly 0, and the target a is
        # also 0 there, so 0*log(0) must not become nan
        loss1a = torch.sum(-a * torch.log(a_hat.clamp_min(1e-9)), dim=(-2, -1))
        # normalize the masked policy CE by the number of attack phases in the
        # batch, not the batch size -- a plain mean divides by every sample and so
        # dilutes (and down-weights) the CE by the non-attacking fraction. clamp
        # the denominator so an all-defense batch does not divide by zero.
        loss1 = torch.sum(loss1a * phase_atk) / phase_atk.sum().clamp_min(1.0)
        loss2 = nn.functional.mse_loss(v_hat, v)
        loss3 = nn.functional.mse_loss(k_hat, k)
        return loss1 + loss2 + loss3, (loss1, loss2, loss3)
