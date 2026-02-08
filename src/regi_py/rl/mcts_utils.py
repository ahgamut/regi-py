from regi_py.core import *
from regi_py.rl.utils import *
from regi_py.logging import DummyLog

#
import random
import time
from collections import defaultdict, UserDict
from dataclasses import dataclass
from numpy.typing import ArrayLike
import numpy as np


def normalize_probs(arr):
    t = np.sum(arr)
    if t != 0:
        arr /= t
    else:
        # this is probably for terminal cases
        # print("throwing")
        arr[-1] = 1.0
    return arr


def enemy_hp_left(game):
    return sum(max(x.hp, 0) for x in game.enemy_pile)


def attack_yieldfail(ind, game, combos):
    if enemy_hp_left(game) == 0:
        return True
    cur_enemy = game.enemy_pile[0]
    if game.get_current_block(cur_enemy) >= cur_enemy.strength:
        # print("yield ok because full block")
        return True
    return random.random() >= 0.4


def defend_throwing(ind, game, combos):
    sel_blk = combos[ind].base_defense
    num_discards = len(combos[ind].parts)
    lower_poss = 0
    for c in combos:
        c_blk = c.base_defense
        c_dsc = len(c.parts)
        if c_dsc < num_discards and c_blk <= sel_blk:
            lower_poss += 1.5
    lower_prob = min(0.9, lower_poss / len(combos))
    return random.random() >= lower_prob


def get_nicer_attacks(game, combos):
    res = []
    if len(combos) < 4:
        return combos
    for ind, c in enumerate(combos):
        if c.bitwise != 0:
            res.append(c)
            continue
        if not attack_yieldfail(ind, game, combos):
            res.append(c)

    return res


def get_nicer_defends(game, combos):
    if len(combos) < 4:
        return combos
    res = []
    N = len(combos)
    for ind in range(N):
        if not defend_throwing(ind, game, combos):
            res.append(combos[ind])
    return res


@dataclass(repr=True, eq=False, slots=True)
class MCTSSample:
    phase: PhaseInfo
    policy: ArrayLike
    value: float
    combos: ArrayLike

    @classmethod
    def eval(cls, phase):
        return MCTSSample(phase, [], 0.0, [])


class PhaseLoader:
    def __init__(self):
        super(PhaseLoader, self).__init__()
        self._counter = 0
        self._forward = dict()
        self._inverse = dict()

    def clear(self):
        self._forward.clear()
        self._inverse.clear()
        self._counter = 0

    def inverse(self, s):
        return self._inverse[s]

    def __getitem__(self, phase):
        pstr = phase.to_string()
        ctr = self._forward.get(pstr, self._counter)
        if ctr == self._counter:
            self._forward[pstr] = self._counter
            self._inverse[self._counter] = phase
            # print(phase, "gets the index", self._counter)
            self._counter += 1
        return ctr


class DupeFailDict(UserDict):
    def __setitem__(self, key, value):
        if key in self.keys():
            if value != self.data[key]:
                raise RuntimeError(f"{key} already exists, points to {self.data[key]}")
        self.data[key] = value


class DupeListDict(UserDict):
    def __setitem__(self, key, value):
        if key in self.keys():
            if value not in self.data[key]:
                # print(f"{key} already exists, has {self.data[key]}")
                self.data[key].append(value)
        else:
            self.data[key] = [value]


class CounterDict(defaultdict):
    def __missing__(self, key):
        return 0


class MCTSTesterStrategy(BaseStrategy):
    __strat_name__ = "mcts-tester"

    def __init__(self, net):
        super(MCTSTesterStrategy, self).__init__()
        self.net = net

    def setup(self, player, game):
        self.net.eval()
        return 0

    def process_phase(self, phase, combos):
        if len(combos) == 0:
            return -1
        probs, v = self.net.predict(MCTSSample.eval(phase))
        cbr = np.zeros(128)
        submap = dict()
        for i, x in enumerate(combos):
            cbr[x.bitwise] = 1
            submap[x.bitwise] = i
        probs *= cbr
        if np.sum(probs) <= 0:
            return -1
        br = random.choices(range(128), weights=probs, k=1)[0]
        if br in submap:
            return submap[br]
        print("sampled invalid move")
        return -1

    def getAttackIndex(self, combos, player, yield_allowed, game):
        return self.process_phase(game.export_phaseinfo(), combos)

    def getDefenseIndex(self, combos, player, damage, game):
        return self.process_phase(game.export_phaseinfo(), combos)


class MCTSTrainerStrategy(BaseStrategy):
    __strat_name__ = "mcts-trainer"

    def __init__(self, coll):
        super(MCTSTrainerStrategy, self).__init__()
        self.coll = coll
        self.net = coll.net
        self.prev_s = None
        self.prev_a = None

    def setup(self, player, game):
        self.net.eval()
        return 0

    def process_phase(self, game, phase, combos):
        if phase.phase_attacking:
            subcombos = get_nicer_attacks(game, combos)
        else:
            subcombos = get_nicer_defends(game, combos)
        br = self.coll.search(phase, subcombos)
        if br == -1:
            return -1
        for i, x in enumerate(combos):
            if x.bitwise == br:
                return i
        # print("sampled invalid move")
        return -1

    def getAttackIndex(self, combos, player, yield_allowed, game):
        ind = self.process_phase(game, game.export_phaseinfo(), combos)
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        ind = self.process_phase(game, game.export_phaseinfo(), combos)
        return ind


class MCTSExplorerStrategy(BaseStrategy):
    __strat_name__ = "mcts-explorer"

    def __init__(self, root_phase):
        super(MCTSExplorerStrategy, self).__init__()
        self.root_phase = root_phase
        self.reroll()

    def reroll(self):
        self.shortcut = None
        self.root_combos = None
        self.next_phases = None
        #
        self.prev_phase = None
        self.prev_a = None
        self.is_recording = True

    def setup(self, player, game):
        return 0

    def mark_combo(self, phase):
        self.next_phases[self.prev_a] = phase

    def update_prev(self, a):
        self.prev_a = a

    def process_phase(self, phase, combos):
        if str(phase) == str(self.root_phase):
            if self.is_recording:
                self.root_combos = combos
                self.next_phases = [None] * len(combos)
            self.prev_phase = phase
        elif str(self.prev_phase) == str(self.root_phase):
            self.mark_combo(phase)
            self.prev_phase = phase
        if self.shortcut is not None:
            ind = self.shortcut
            self.shortcut = None
        else:
            ind = random.choice(range(len(combos)))
        self.update_prev(ind)
        return ind

    def getAttackIndex(self, combos, player, yield_allowed, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        return ind


def get_expansion_at(root_phase):
    log = DummyLog()
    tmp = GameState(log)
    exp_strat = MCTSExplorerStrategy(root_phase)
    for i in range(root_phase.num_players):
        tmp.add_player(exp_strat)

    tmp._init_phaseinfo(root_phase)
    tmp.start_loop()
    root_combos = exp_strat.root_combos
    exp_strat.is_recording = False

    for i in range(len(root_combos)):
        exp_strat.shortcut = i
        tmp._init_phaseinfo(root_phase)
        tmp.start_loop()
        if exp_strat.next_phases[i] is None:
            exp_strat.next_phases[i] = tmp.export_phaseinfo()

    next_phases = exp_strat.next_phases
    assert len(next_phases) == len(root_combos)
    for x in next_phases:
        assert x is not None

    return next_phases, root_combos


def quick_game_value(root_phase, relative_diff=False):
    log = DummyLog()
    tmp = GameState(log)
    exp_strat = RandomStrategy()
    for i in range(root_phase.num_players):
        tmp.add_player(exp_strat)
    tmp._init_phaseinfo(root_phase)
    tmp.start_loop()
    if relative_diff:
        vstart = enemy_hp_left(root_phase)
    else:
        vstart = 360
    vend = enemy_hp_left(tmp.export_phaseinfo())
    val = (vstart - vend) / 360
    return val


class MCTSLog(DummyLog):
    def __init__(self, coll):
        super().__init__()
        self.coll = coll
        self.memories = []

    ####
    def endgame(self, reason, game):
        end_phase = game.export_phaseinfo()
        self.coll.search(end_phase, [])
        self.coll.reset_sim()
