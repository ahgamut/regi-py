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
