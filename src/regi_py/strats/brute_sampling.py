import random
import numpy as np

#
from regi_py.core import BaseStrategy
from regi_py.core import RandomStrategy
from regi_py.strats.recommender import RecommenderMixin
from regi_py.strats.phase_utils import *


class BruteSamplingStrategy(BaseStrategy, RecommenderMixin):
    __strat_name__ = "brute"

    def __init__(self, iterations=128, num_recos=5):
        super().__init__()
        self.__strat_name__ = f"brute-{iterations}"
        self.iterations = iterations
        self.num_recos = num_recos

    def setup(self, player, game):
        return 0

    def getRedirectIndex(self, player, game):
        offset = random.randint(1, game.num_players - 1)
        return (game.active_player + offset) % game.num_players

    def process_moves(self, root_phase, combos):
        next_phases, next_combos = get_expansion_at(root_phase, trim=True)
        B = self.iterations

        N = len(next_combos)
        vals = [0] * N
        val_cur = np.zeros(B, dtype=np.float32)

        for i in range(N):
            for b in range(B):
                val_cur[b] = quick_game_value(
                    next_phases[i], strat_klass=RandomStrategy, relative_diff=True
                )
            # print(val_cur)
            if root_phase.phase_attacking:
                yield_penalty = random.random() * int(next_combos[i].bitwise == 0)
                vals[i] = np.quantile(val_cur, 0.9) - yield_penalty
            else:
                def_penalty = defend_throwing(i, root_phase, next_combos, score_only=True)
                vals[i] = np.quantile(val_cur, 0.9) - 0.5 * def_penalty

        val_arr = np.array(vals)
        return next_combos, val_arr

    def get_best_move(self, root_phase, combos):
        moves, scores = self.process_moves(root_phase, combos)
        best = int(np.argmax(scores))

        best_move = next_combos[best]
        for ind, c in enumerate(combos):
            if c.bitwise == best_move.bitwise:
                return ind
        return -1

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        #
        root_phase = game.export_phaseinfo()
        try:
            ind = self.get_best_move(root_phase, combos)
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
        root_phase = game.export_phaseinfo()
        try:
            ind = self.get_best_move(root_phase, combos)
        except Exception as e:
            print("failed to process moves", e)
            ind = random.randint(0, len(combos) - 1)
        if defend_throwing(ind, game, combos):
            print("this defend is a throw", ind, combos[ind])
        return ind

    def getRecommendedMoves(self, phase, combos):
        moves, scores = self.process_moves(phase, combos)
        sinds = np.argsort(scores)[::-1]
        nr = min(self.num_recos, len(scores))
        recos = [moves[int(x)] for x in sinds[:nr]]
        return recos
