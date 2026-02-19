from regi_py.core import BaseStrategy
from regi_py.strats.phase_utils import *
import random


class TrimmedRandomStrategy(BaseStrategy):
    __strat_name__ = "trim-random"

    def setup(self, player, game):
        return 0

    def getRedirectIndex(self, player, game):
        offset = random.randint(1, game.num_players - 1)
        return (game.active_player + offset) % game.num_players

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        ind = random.randint(0, len(combos) - 1)
        if attack_yieldfail(ind, game, combos):
            return -1
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        ind = random.randint(0, len(combos) - 1)
        if defend_throwing(ind, game, combos):
            return -1
        return ind
