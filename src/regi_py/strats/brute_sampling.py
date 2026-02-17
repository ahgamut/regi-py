import random
import numpy as np

#
from regi_py.core import BaseStrategy
from regi_py.strats.mcts_explorer import get_expansion_at
from regi_py.strats.phase_utils import *
from regi_py.strats.trim_random import quick_game_value


class BruteSamplingStrategy(BaseStrategy):
    __strat_name__ = "brute"

    def setup(self, player, game):
        return 0

    def getRedirectIndex(self, player, game):
        offset = random.randint(1, game.num_players - 1)
        return (game.active_player + offset) % game.num_players

    def process_moves(self, game, combos):
        root_phase = game.export_phaseinfo()
        next_phases, next_combos = get_expansion_at(root_phase, trim=True)
        B = 128

        N = len(next_combos)
        vals = [0] * N
        val_cur = np.zeros(B, dtype=np.float32)

        for i in range(N):
            for b in range(B):
                val_cur[b] = quick_game_value(next_phases[i], relative_diff=True)
            # print(val_cur)
            if root_phase.phase_attacking:
                yield_penalty = random.random() * int(next_combos[i].bitwise == 0)
                vals[i] = np.quantile(val_cur, 0.9) - yield_penalty
            else:
                def_penalty = defend_throwing(i, game, next_combos, score_only=True)
                vals[i] = np.quantile(val_cur, 0.9) - 0.5 * def_penalty

        val_arr = np.array(vals)
        best = int(np.argmax(val_arr))

        best_move = next_combos[best]
        for ind, c in enumerate(combos):
            if c.bitwise == best_move.bitwise:
                return ind
        return -1

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        #
        try:
            ind = self.process_moves(game, combos)
        except Exception as e:
            print("failed to process moves", e)
            ind = random.randint(0, len(combos) - 1)
        if combos[ind].bitwise == 0 and attack_yieldfail(ind, game, combos):
            print("this yield is randomly a fail", ind, combos[ind])
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        #
        try:
            ind = self.process_moves(game, combos)
        except Exception as e:
            print("failed to process moves", e)
            ind = random.randint(0, len(combos) - 1)
        if defend_throwing(ind, game, combos):
            print("this defend is a throw", ind, combos[ind])
        return ind
