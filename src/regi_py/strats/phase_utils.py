from regi_py.core import *
from regi_py.logging import DummyLog
import random


def enemy_hp_left(game):
    return sum(max(x.hp, 0) for x in game.enemy_pile)


def attack_yieldfail(ind, game, combos):
    if len(combos) < 4:
        return False
    if enemy_hp_left(game) == 0:
        return False
    cur_enemy = game.enemy_pile[0]
    if game.get_current_block(cur_enemy) >= cur_enemy.strength:
        # print("yield ok because full block")
        return False
    return random.random() <= 0.4


def defend_throwing(ind, game, combos):
    if len(combos) < 4:
        return False
    sel_blk = combos[ind].base_defense
    num_discards = len(combos[ind].parts)
    lower_poss = 0
    for c in combos:
        c_blk = c.base_defense
        c_dsc = len(c.parts)
        if c_dsc < num_discards and c_blk <= sel_blk:
            lower_poss += 1.5
    lower_prob = min(0.9, lower_poss / len(combos))
    return random.random() <= lower_prob


def get_nicer_attacks(game, combos):
    res = []
    if len(combos) < 4:
        return combos
    for ind, c in enumerate(combos):
        if c.bitwise != 0:
            res.append(c)
            continue
        if random.random() > 0.4:
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
