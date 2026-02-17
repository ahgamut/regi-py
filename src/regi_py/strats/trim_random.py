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


def quick_game_value(root_phase, relative_diff=False):
    log = DummyLog()
    tmp = GameState(log)
    exp_strat = TrimmedRandomStrategy()
    for i in range(root_phase.num_players):
        tmp.add_player(exp_strat)
    tmp._init_phaseinfo(root_phase)
    tmp.start_loop()
    end_phase = tmp.export_phaseinfo()
    if end_phase.game_endvalue == 1:
        return 2
    #
    if relative_diff:
        vstart = enemy_hp_left(root_phase)
    else:
        vstart = 360
    vend = enemy_hp_left(end_phase)
    val = (vstart - vend) / 360
    return val
