from regi_py.core import BaseStrategy
from regi_py.strats.phase_utils import *
import random


class SubsetRandomStrategy(BaseStrategy):
    __strat_name__ = "sub-random"

    def setup(self, player, game):
        return 0

    def getRedirectIndex(self, player, game):
        offset = random.randint(1, game.num_players - 1)
        return (game.active_player + offset) % game.num_players

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        sub = get_nonbad_attacks(game, combos)
        move = random.choice(sub)
        return indexify(move, combos)

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        sub = get_nonbad_defends(game, combos)
        move = random.choice(sub)
        return indexify(move, combos)
