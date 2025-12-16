from regi_py.core import BaseStrategy
import random


class DummyStrategy(BaseStrategy):
    __strat_name__ = "dummy"

    def setup(self, player, game):
        return 0

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        # print("available attacks: ", combos)
        return random.randint(0, len(combos) - 1)

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        # print("available defenses: ", combos)
        return random.randint(0, len(combos) - 1)
