from regi_py.core import *
from regi_py.rl.utils import *
from regi_py.strats.phase_utils import *
from regi_py.strats.mcts_explorer import get_expansion_at
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
