"""``CandidateBaseNet`` -- the contract every AlphaDouZero (ADZ) architecture
satisfies.

Where an AZ ``BaseNet`` emits a fixed ``(56, 22)`` policy grid (which *is* the
attack ``ComboTable``, so defense has no head), an ADZ net models EVERY card-play
decision -- attack AND defense -- as a masked softmax over the phase's *ragged*
list of offered subsets (DouZero-style ``(state, action)`` scoring wrapped in
AlphaZero search). The node (`adz_explorer.ADZNode`) stays agnostic to how a net
encodes a candidate subset: it stores the offered ``bitwise``s, a net-agnostic
per-candidate feature block, and the visit policy, and each net decides its own
membership encoding via the ``_assemble_membership`` hook (multi-hot first;
index+mask/pooled later -- adding one must NOT touch the node or this base).

Shared here (driven by ``_assemble_membership`` + ``features.py``):
  - state featurization: one token per card over the history window
    (``features.fuse_card_tokens``), identical to the card-token AZ nets.
  - candidate targets/aux, all padded/masked to ``MAX_CANDIDATES``:
    ``cand_feats`` (semantic per-candidate block), ``cand_mask``, ``policy``
    (visit fractions), ``keepyness`` (derived CFR aux), ``value``, ``attacking``.
  - ``predict`` / ``tensorify_predict`` / ``tensorify_training`` / ``calculate_loss``.

A concrete net overrides ``__mname__``, ``TRAIN_FIELDS``, ``_assemble_membership``
and ``forward``; everything else is inherited.
"""
from regi_py.core import MAX_CARDS_IN_GAME
from regi_py.rl import features

import numpy as np
import torch
import torch.nn as nn


class CandidateBaseNet(nn.Module):
    # ---- class contract each concrete net sets ----
    __mname__ = "adzbase"        # checkpoint tag + strat-name suffix; must be unique
    max_history = 8              # history window length (state half)
    # max offered subsets at any decision: a 7-card hand yields <= 2^7 = 128
    # defenses; attacks are far smaller. ShardBuffer stacks rows uniformly across
    # shards, so every row must be this fixed width.
    MAX_CANDIDATES = 128
    CAND_FEATURE_DIM = features.CAND_FEATURE_DIM
    TRAIN_FIELDS = ()            # input columns THEN target columns, in shard order

    def __init__(self):
        super().__init__()
        self.device = "cpu"

    # ---- subclass implements ----
    def forward(self, data):
        """dict of input tensors -> ``(value (N,1), cand_logits (N,MAX_CANDIDATES)
        with -inf on padded/masked slots, keepy (N,56))``."""
        raise NotImplementedError

    @classmethod
    def _assemble_membership(cls, padded_bitwises):
        """Encode candidate membership for this net. ``padded_bitwises`` is a list
        of length ``MAX_CANDIDATES`` of the offered combos' ``bitwise`` masks
        (real candidates first, padded with 0). Returns a dict of ``(1, ...)``
        tensors -- the ONLY per-encoding field(s). Multi-hot returns
        ``{"cand_members": (1, MAX_CANDIDATES, 56)}``; an index+mask net would
        return ``{"cand_idx": ..., "cand_partmask": ...}`` instead. Overridden per
        architecture; the node/record are untouched when a new encoding is added.
        """
        raise NotImplementedError

    # ---- shared state + candidate featurization ----
    @classmethod
    def _state_tokens(cls, history, perspective, window):
        """One token per card over the history window: ``(1, 56, window*33)``.
        Identical layout to the card-token AZ nets (``fuse_card_tokens``); the
        used-pile trajectory is carried here as a first-class *state* input."""
        loc, usp, cap = features.raw_window_arrays(history, perspective, window)
        tok = features.fuse_card_tokens(loc, usp, cap)
        return torch.from_numpy(tok).unsqueeze(0)

    @classmethod
    def _pack_candidates(cls, cand_feats, bitwises):
        """Pad the ragged candidate block to ``MAX_CANDIDATES`` and encode
        membership. ``cand_feats`` is ``(K, F)`` (from ``candidate_semantics``);
        ``bitwises`` is the aligned length-``K`` list of ``Combo.bitwise``. Returns
        a dict with ``cand_feats (1,MC,F)``, ``cand_mask (1,MC)`` plus the net's
        membership tensor(s). Shared by ``tensorify_predict`` and
        ``tensorify_training`` so the two paths stay byte-identical."""
        MC = cls.MAX_CANDIDATES
        K = len(bitwises)
        if K > MC:
            raise ValueError(f"{K} candidates exceed MAX_CANDIDATES={MC}")
        feats = np.zeros((MC, cls.CAND_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(MC, dtype=np.float32)
        padded_bw = [0] * MC
        if K:
            feats[:K] = cand_feats
            mask[:K] = 1.0
            padded_bw[:K] = [int(b) for b in bitwises]
        out = {
            "cand_feats": torch.from_numpy(feats).unsqueeze(0),
            "cand_mask": torch.from_numpy(mask).unsqueeze(0),
        }
        out.update(cls._assemble_membership(padded_bw))
        return out

    # ---- shared tensorify / predict / loss ----
    @classmethod
    def tensorify_predict(cls, history, offered_combos, phase, perspective=None, window=None):
        """Input dict for one live decision: state tokens from ``history`` plus the
        padded candidate block for ``offered_combos`` (semantics computed live from
        the ``phase`` -- both are available at self-play export and at inference)."""
        if window is None:
            window = cls.max_history
        if perspective is None:
            perspective = history[-1].active_player
        cand_feats = features.candidate_semantics(phase, offered_combos)
        bitwises = [c.bitwise for c in offered_combos]
        data = cls._pack_candidates(cand_feats, bitwises)
        data["tokens"] = cls._state_tokens(history, perspective, window)
        return data

    @classmethod
    def tensorify_training(cls, infos):
        """Batch a list of ``ADZNodeInfo`` records into the training dict. Every
        ``TRAIN_FIELDS`` column is padded/masked to ``MAX_CANDIDATES``; the record
        stores ``cand_feats`` verbatim (defense combos can't be rebuilt from a bare
        ``bitwise``) and ``keepyness`` is DERIVED here from the visit policy."""
        N = len(infos)
        window = len(infos[0].history)
        MC = cls.MAX_CANDIDATES
        value = torch.zeros((N, 1))
        attacking = torch.zeros((N, 1))
        keepyness = torch.ones((N, MAX_CARDS_IN_GAME))
        policy = torch.zeros((N, MC))
        input_cols = None
        for i, info in enumerate(infos):
            cur_phase = info.history[-1]
            perspective = cur_phase.active_player
            cand_feats = np.asarray(info.cand_feats, dtype=np.float32)
            packed = cls._pack_candidates(cand_feats, info.candidates)
            packed["tokens"] = cls._state_tokens(info.history, perspective, window)
            if input_cols is None:
                input_cols = {k: [] for k in packed}
            for k, v in packed.items():
                input_cols[k].append(v)
            K = len(info.candidates)
            if K:
                policy[i, :K] = torch.from_numpy(np.asarray(info.policy, dtype=np.float32))
            keepyness[i, :] = torch.from_numpy(
                features.keepy_marginal(info.candidates, info.policy)
            )
            value[i, 0] = float(info.value)
            attacking[i, 0] = float(info.attacking)
        result = {k: torch.cat(v, dim=0) for k, v in input_cols.items()}
        result["policy"] = policy
        result["value"] = value
        result["keepyness"] = keepyness
        result["attacking"] = attacking
        return result

    def predict(self, history, offered_combos, phase, perspective=None):
        """Inference contract: ``(value, priors)`` where ``priors`` is a
        ``len(offered_combos)`` numpy array of a softmax over the REAL (unpadded)
        candidates, aligned to ``offered_combos``. Empty offer -> empty priors."""
        with torch.inference_mode():
            data = type(self).tensorify_predict(
                history, offered_combos, phase, perspective, self.max_history
            )
            value, cand_logits, _ = self.forward(data)
            v_hat = float(value.detach().cpu().numpy()[0, 0])
            K = len(offered_combos)
            if K == 0:
                return v_hat, np.zeros(0, dtype=np.float32)
            logits = cand_logits[0, :K]
            priors = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return v_hat, priors.astype(np.float32)

    def calculate_loss(self, data, y_hat):
        """Masked policy CE over the candidate axis + value MSE + keepy MSE.
        Returns ``(total, (policy, value, keepy))`` -- same 3-tuple shape as the AZ
        loss, so ``run_epoch`` and the trainer's per-head logging are shared."""
        value, cand_logits, keepy = y_hat
        policy = data["policy"]
        mask = data["cand_mask"]
        # forward emits -inf on padded slots. Feeding -inf straight into
        # log_softmax is unsafe for BACKWARD: a fully-masked row (a terminal
        # self-play record has zero candidates) makes the log_softmax jacobian
        # inf, and inf * 0 (the zero upstream grad) = nan, which would poison the
        # whole batch's gradient. Rebuild finite logits from the mask instead --
        # real slots keep their logit, pads get dtype.min -- so log_softmax is
        # finite everywhere; a fully-masked row is then uniform and, times its
        # all-zero policy target, contributes 0 to the loss with 0 gradient.
        neg = torch.finfo(cand_logits.dtype).min
        safe_logits = torch.where(mask > 0, cand_logits, torch.full_like(cand_logits, neg))
        log_probs = torch.log_softmax(safe_logits, dim=-1)
        ce = -(policy * log_probs).sum(dim=-1)
        # every decision (attack AND defense) carries a policy, so normalize by the
        # number of decisions in the batch (all rows) -- the AZ attack-count mask
        # is gone because there is no longer an attack-only head
        loss1 = ce.mean()
        loss2 = nn.functional.mse_loss(value, data["value"])
        loss3 = nn.functional.mse_loss(keepy, data["keepyness"])
        return loss1 + loss2 + loss3, (loss1, loss2, loss3)
