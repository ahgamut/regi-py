from regi_py.core import PhaseInfo, LocationInfo, ComboTable, Card
from regi_py.core import MAX_CARDS_IN_GAME, MAX_LOCATIONS, MAX_PLAYED_STATUS
from regi_py.rl.az_explorer import AZNodeInfo
from regi_py.rl.utils import *
from regi_py.rl.subnets import (
    Conv1dBlock,
    Conv2dBlock,
    WidthCrossAttention,
)

#
import numpy as np
import torch
import torch.nn as nn

# per-card capability channels: [attack_capability, defense_capability]
CAP_CHANNELS = 2
# scale factor keeping capabilities roughly in [-1, 1]: a King has 40 HP -> -1.0
# and deals 20 base damage -> -0.5
CAP_SCALE = 40.0
# flattened trunk width for the value/keepy heads: the trunk now lives on the
# used-pile grid (card axis x played-status axis), so 56 x 22
_TRUNK_FLAT = MAX_CARDS_IN_GAME * int(MAX_PLAYED_STATUS)

# static raw per-card strength by location; enemy-pile cards override this each
# phase (their HP / base damage), so only the non-enemy part is precomputed
_STRENGTH = np.zeros(MAX_CARDS_IN_GAME, dtype=np.float32)
for _loc in range(MAX_CARDS_IN_GAME):
    try:
        _STRENGTH[_loc] = Card.from_location(_loc).strength
    except Exception:
        pass


def card_capabilities(phase):
    """Per-card ``(attack, defense)`` capability for ``phase``, on the 56-location
    axis. Shape ``(MAX_CARDS_IN_GAME, CAP_CHANNELS)``, scaled by ``1/CAP_SCALE``.

    A card *not* in the enemy pile contributes its own (non-negative) strength to
    both channels. A card *in* the enemy pile is a target, encoded negatively:
    attack = ``-max(0, current HP)``, defense = ``-(base damage it deals)``.
    """
    attack = _STRENGTH.copy()
    defense = _STRENGTH.copy()
    for enemy in phase.enemy_pile:
        loc = enemy.location
        attack[loc] = -max(0, enemy.hp)
        defense[loc] = -enemy.strength
    caps = np.empty((MAX_CARDS_IN_GAME, CAP_CHANNELS), dtype=np.float32)
    caps[:, 0] = attack
    caps[:, 1] = defense
    caps /= CAP_SCALE
    return caps


# content-keyed caches (by phase.to_string()) so the rolling history window and
# sibling MCTS nodes don't re-tensorize the same phase repeatedly
_CACHE_CAP = 8192
_LOC_CACHE = {}  # (phase_str, perspective) -> np (56, 9)
_USP_CACHE = {}  # phase_str -> np (56, 22)
_CAP_CACHE = {}  # phase_str -> np (56, CAP_CHANNELS)


def _cache_put(cache, key, val):
    if len(cache) >= _CACHE_CAP:
        cache.clear()
    cache[key] = val
    return val


def _location_array(phase, perspective):
    key = (phase.to_string(), perspective)
    a = _LOC_CACHE.get(key)
    if a is None:
        loca0 = np.array(LocationInfo.from_current(phase, perspective), dtype=np.float32)
        a = _cache_put(_LOC_CACHE, key, loca0 / loca0.sum(axis=1, keepdims=True))
    return a


def _used_pile_array(phase):
    key = phase.to_string()
    a = _USP_CACHE.get(key)
    if a is None:
        a = _cache_put(_USP_CACHE, key, np.array(ComboTable.from_phase(phase), dtype=np.float32))
    return a


def _capability_array(phase):
    key = phase.to_string()
    a = _CAP_CACHE.get(key)
    if a is None:
        a = _cache_put(_CAP_CACHE, key, card_capabilities(phase))
    return a


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = Conv2dBlock(channels=(64, 8, 1), shapes=(3, 1), paddings=(1, 0))
        self.net2 = nn.Linear(in_features=_TRUNK_FLAT, out_features=1)
        self.ac = nn.Sigmoid()

    def forward(self, x):
        x = self.net1(x).reshape(x.shape[0], -1)
        x = self.net2(x)
        x = self.ac(x)
        return x


class KeepyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = Conv2dBlock(channels=(64, 8, 1), shapes=(3, 1), paddings=(1, 0))
        self.net2 = nn.Linear(in_features=_TRUNK_FLAT, out_features=MAX_CARDS_IN_GAME)
        self.ac = nn.Tanh()

    def forward(self, x):
        x = self.net1(x).reshape(x.shape[0], -1)
        x = self.net2(x)
        x = self.ac(x)
        return x


class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        n = MAX_CARDS_IN_GAME
        # trunk (N, 64, 56, 22) -> one logit per (card, played_status) cell, with
        # the 56 x 22 grid preserved throughout
        self.net1 = Conv2dBlock(channels=(64, 16, 1), shapes=(3, 1), paddings=(1, 0))
        # mix along the played-status axis (56 cards as channels), width preserved
        self.net2 = Conv1dBlock(channels=(n, n, n), shapes=(3, 3), paddings=(1, 1))
        # fuse per-card keepyness into the per-card action logits; 56 % 8 == 0
        self.wca = WidthCrossAttention(channels=MAX_CARDS_IN_GAME, heads=8)
        # additive softmax mask over the flattened (56 x 22) action grid: 0 on
        # structurally-valid (location, played_status) cells, -inf on impossible
        # ones. Built once here so the softmax just adds it (no per-forward fill).
        valid = np.array(ComboTable.all_entries(), dtype=np.float32)  # (56, 22)
        add_mask = np.where(valid == 0, -np.inf, 0.0).astype(np.float32).reshape(1, -1)
        self.register_buffer("invalid_mask", torch.from_numpy(add_mask))

    def forward(self, x0, k):
        x = self.net1(x0)  # (N, 1, 56, 22)
        x = x.reshape(x0.shape[0], MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)  # (N, 56, 22)
        x = self.net2(x)  # (N, 56, 22)
        x = x.reshape(x.shape[0], MAX_CARDS_IN_GAME, 1, MAX_PLAYED_STATUS)
        k2 = k.reshape(x.shape[0], MAX_CARDS_IN_GAME, 1, 1)
        logits = self.wca(x, k2).reshape(-1, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)
        # masked softmax over the whole (56 x 22) action grid
        n = logits.shape[0]
        flat = torch.softmax(logits.reshape(n, -1) + self.invalid_mask, dim=-1)
        return flat.reshape(-1, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)


class CombineNet(nn.Module):
    def __init__(self, channels=32, reduction=4):
        super().__init__()
        self.wca1 = WidthCrossAttention(channels=channels, heads=4)
        self.wca2 = WidthCrossAttention(channels=channels, heads=4)
        self.net = Conv2dBlock(
            channels=(channels, 64, 64, 64, 64, 64),
            shapes=(1, 3, 3, 3, 3),
            paddings=(0, 1, 1, 1, 1),
        )

    def forward(self, usp, loc, cap):
        y1 = self.wca1(usp, loc)
        y2 = self.wca2(y1, cap)
        y3 = self.net(y2)
        return y3


class BasicNet(nn.Module):
    __mname__ = "basic"

    # single source of truth for the training tensor field order, shared by
    # tensorify_training / ShardBuffer / run_epoch
    TRAIN_FIELDS = (
        "location",
        "used_pile",
        "capability",
        "value",
        "keepyness",
        "atk_probs",
        "attacking",
    )

    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.max_history = 8
        #
        self.loc_net = Conv2dBlock(
            channels=(self.max_history, 32),
            shapes=(1,),
            paddings=(0,),
        )
        self.usp_net = Conv2dBlock(
            channels=(self.max_history, 32),
            shapes=(1,),
            paddings=(0,),
        )
        self.cap_net = Conv2dBlock(
            channels=(self.max_history, 32),
            shapes=(1,),
            paddings=(0,),
        )

        self.combiner = CombineNet(channels=32)
        self.v_net = ValueNet()
        self.k_net = KeepyNet()
        self.a_net = ActionNet()

    def forward(self, data):
        x1 = self.usp_net(data["used_pile"])
        x2 = self.loc_net(data["location"])
        x3 = self.cap_net(data["capability"])
        x = self.combiner(x1, x2, x3)
        v = self.v_net(x)
        k = self.k_net(x)
        a = self.a_net(x, k)
        return v, k, a

    def calculate_loss(self, y, y_hat, phase_atk):
        v, k, a = y
        v_hat, k_hat, a_hat = y_hat
        # clamp inside log: masked cells make a_hat exactly 0, and the target a is
        # also 0 there, so 0*log(0) must not become nan
        loss1a = torch.sum(-a * torch.log(a_hat.clamp_min(1e-9)), dim=(-2, -1))
        loss1 = torch.mean(loss1a * phase_atk)
        loss2 = nn.functional.mse_loss(v_hat, v)
        loss3 = nn.functional.mse_loss(k_hat * k, k)
        return loss1 + loss2 + loss3

    def predict(self, history, perspective=None):
        data = BasicNet.tensorify_phases(history, perspective, self.max_history)
        v_hat0, k_hat0, a_hat0 = self.forward(data)
        v_hat = float(v_hat0.detach().cpu().numpy()[0, 0])
        k_hat = k_hat0.detach().cpu().numpy()[0, :]
        a_hat = a_hat0.detach().cpu().numpy()[0, 0, :, :]
        return v_hat, k_hat, a_hat

    @staticmethod
    def tensorify_phases(history, perspective=None, window=8):
        result = {
            "location": torch.zeros((1, window, MAX_CARDS_IN_GAME, MAX_LOCATIONS)),
            "used_pile": torch.zeros((1, window, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)),
            "capability": torch.zeros((1, window, MAX_CARDS_IN_GAME, CAP_CHANNELS)),
        }
        if perspective is None:
            perspective = history[-1].active_player
        #
        for j in range(window):
            phase = history[j]
            result["location"][0, j] = torch.from_numpy(_location_array(phase, perspective))
            result["used_pile"][0, j] = torch.from_numpy(_used_pile_array(phase))
            result["capability"][0, j] = torch.from_numpy(_capability_array(phase))
        return result

    @staticmethod
    def tensorify_training(infos):
        N = len(infos)
        window = len(infos[0].history)
        result = {
            "location": torch.zeros((N, window, MAX_CARDS_IN_GAME, MAX_LOCATIONS)),
            "used_pile": torch.zeros((N, window, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)),
            "capability": torch.zeros((N, window, MAX_CARDS_IN_GAME, CAP_CHANNELS)),
            "value": torch.zeros((N, 1)),
            "keepyness": torch.ones((N, MAX_CARDS_IN_GAME)),
            "atk_probs": torch.zeros((N, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)),
            "attacking": torch.zeros((N, 1)),
        }
        #
        for i in range(N):
            info = infos[i]
            cur_phase = info.history[-1]
            result["value"][i, 0] = info.value
            result["attacking"][i, 0] = cur_phase.phase_attacking
            result["keepyness"][i, :] = torch.from_numpy(info.keepyness)
            result["atk_probs"][i, 0] = torch.from_numpy(info.atk_probs)
            #
            perspective = cur_phase.active_player
            for j in range(window, 0, -1):
                phase = info.history[-j]
                result["location"][i, -j] = torch.from_numpy(_location_array(phase, perspective))
                result["used_pile"][i, -j] = torch.from_numpy(_used_pile_array(phase))
                result["capability"][i, -j] = torch.from_numpy(_capability_array(phase))

        return result
