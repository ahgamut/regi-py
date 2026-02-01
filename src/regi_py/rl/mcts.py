from regi_py.core import *
from regi_py.rl.mcts_utils import *
from regi_py.logging import DummyLog

#
import random
import time
from collections import defaultdict, UserDict
from dataclasses import dataclass
from numpy.typing import ArrayLike
import numpy as np

# https://mcts.ai/code/python.html

# https://github.com/suragnair/alpha-zero-general
# f1c98a9b37e4b3bdddd4595966a7398829795817


class MCTSCollector:
    def __init__(self, net, puct, epsilon=1e-8, bound=30):
        self.bound = bound
        self.net = net
        self.puct = puct
        self.epsilon = epsilon
        self.phases = PhaseLoader()  # [PhaseInfo] -> s
        #
        self.f_edges = DupeListDict()  # [(s, a)] -> s'
        self.b_edges = DupeListDict()  # [s'] -> (s, a)
        #
        self.Q = dict()  # [(s,a)] -> q
        self.N0 = CounterDict()  # [s] -> int
        self.N1 = CounterDict()  # [(s,a)] -> int
        self.P = dict()  # [s] -> probability vector
        self.E = DupeFailDict()  # [s] -> ??? (only for endgame?)
        #
        self.depth = dict()  # [s] -> phase_count (basically)
        self.C = DupeFailDict()  # [s] -> combos
        self.vals = DupeFailDict()  # [s] -> v (filled backwards?)
        #
        self.prev_s = None
        self.prev_a = None

    def __len__(self):
        return len(self.N0)

    def reset_sim(self):
        self.prev_s = None
        self.prev_a = None

    def clear(self):
        self.phases.clear()
        #
        self.f_edges.clear()
        self.b_edges.clear()
        #
        self.Q.clear()
        self.N0.clear()
        self.N1.clear()
        self.P.clear()
        self.E.clear()
        #
        self.depth.clear()
        self.C.clear()
        self.vals.clear()
        #
        self.reset_sim()

    def search(self, phase, combos):
        cur_s = self.phases[phase]
        self._connect(self.prev_s, self.prev_a, cur_s)
        a = self._search(cur_s, phase, combos)
        self.prev_s = cur_s
        self.prev_a = a
        # print(f"picked ({self.prev_s}, {self.prev_a})")
        return a

    def calc_policy(self, s):
        den = self.N0[s] + self.epsilon
        arr = np.zeros(len(self.C[s]), dtype=np.float32)
        if len(arr) > 0:
            for a in range(len(self.C[s])):
                arr[a] = self.N1[(s, a)] / den
            if self.depth[s] > self.bound:
                best = np.argmax(arr)
                arr *= 0
                arr[best] = 1.0
        # print(f"policy for {s} is", arr)
        return arr

    def _search(self, s, phase, combos):
        if s not in self.C:
            cbr = np.zeros(128)
            for x in combos:
                cbr[x.bitwise] = 1
            self.C[s] = cbr
        #
        if s not in self.E:
            self.E[s] = phase.game_endvalue
        if phase.game_endvalue != 0:
            # print("ending at", s, "with", phase.game_endvalue)
            self.vals[s] = phase.game_endvalue
            self.update_backwards(s)
            return -1
        if s not in self.P:
            # print("getting probs for", s)
            # normalized probs
            self.P[s], v = self.net.predict(MCTSSample.eval(phase))
            self.P[s] *= self.C[s]
            t = np.sum(self.P[s])
            if t != 0:
                self.P[s] /= t
            self.vals[s] = v
            self.N0[s] = 0
            self.update_backwards(s)
            return random.choices(range(128), weights=self.P[s], k=1)[0]

        # ucb
        cur_best = -float("inf")
        best_act = 0

        # pick the action with the highest upper confidence bound
        for a in range(len(combos)):
            prob_a = self.P[s][a]
            if (s, a) in self.Q:
                u1 = self.Q[(s, a)]
                u2 = self.puct * prob_a * np.sqrt(self.N0[s]) / (1 + self.N1[(s, a)])
                u = u1 + u2
            else:  # Q = 0 ?
                u = self.puct * prob_a * np.sqrt(self.N0[s] + self.epsilon)

            if u > cur_best:
                cur_best = u
                best_act = a

        # print(f"best_act from {s} is {best_act} (u={cur_best})")
        return best_act

    def _connect(self, prev_s, prev_a, cur_s):
        if prev_s is None:
            self.depth[cur_s] = 0
            return
        # print("adding edge", (prev_s, prev_a), "->", cur_s)
        self.f_edges[(prev_s, prev_a)] = cur_s
        self.b_edges[cur_s] = (prev_s, prev_a)  # ouch
        self.depth[cur_s] = self.depth[prev_s] + 1

    def update_backwards(self, next_s):
        v = self.vals[next_s]

        def set_qn(s0):
            for x in self.b_edges.get(s0, []):
                # print("updating", x, "from", s0)
                s, a = x
                if s is None:
                    return
                if (s, a) in self.Q:
                    q1 = self.N1[(s, a)] * self.Q[(s, a)] + v
                    q2 = 1 + self.N1[(s, a)]
                    self.Q[(s, a)] = q1 / q2
                    self.N1[(s, a)] += 1
                else:
                    self.Q[(s, a)] = v
                    self.N1[(s, a)] = 1

                self.N0[s] += 1
                set_qn(s)

        set_qn(next_s)

    def rewardize(self, end_phase):
        r1 = dict()
        s_end = self.phases[end_phase]
        pi_end = self.calc_policy(s_end)
        r1[s_end] = MCTSSample(end_phase, pi_end, self.E[s_end], self.C[s_end])

        def set_egv(ind, next_ind, res):
            if ind not in res:
                # print(ind, "gets the value: ", res[next_ind].value)
                phase = self.phases.inverse(ind)
                policy = self.calc_policy(ind)
                reward = MCTSSample(phase, policy, res[next_ind].value, self.C[ind])
                res[ind] = reward
            for x in self.b_edges.get(ind, []):
                s, a = x
                if s is None:
                    return
                set_egv(s, ind, res)

        set_egv(s_end, None, r1)
        return r1


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

    def process_phase(self, phase, combos):
        br = self.coll.search(phase, combos)
        if br == -1:
            return -1
        for i, x in enumerate(combos):
            if x.bitwise == br:
                return i
        # print("sampled invalid move")
        return -1

    def getAttackIndex(self, combos, player, yield_allowed, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        if ind == -1 or len(combos) < 4:
            return ind
        if ind == 0 and random.random() < 0.4:
            # print("randomly fail yield")
            return -1
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        if ind == -1 or len(combos) < 4:
            return ind
        # 
        sel_blk = combos[ind].base_defense
        num_discards = len(combos[ind].parts)
        lower_poss = 0
        for i, c in enumerate(combos):
            c_blk = c.base_defense
            c_dsc = len(c.parts)
            if c_dsc < num_discards and c_blk <= sel_blk:
                lower_poss += 1
        if random.random() < (lower_poss / len(combos)):
            # print("randomly fail blk:", combos[ind], lower_poss)
            return -1
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
        # print(self.coll.N1)
        self.memories = list(self.coll.rewardize(end_phase).values())
        self.coll.reset_sim()


class MCTS:
    def __init__(self, num_players, net, puct=0.1, N=1000):
        self.N = N
        self.examples = []
        self.played_games = set()
        #
        self.num_players = num_players
        self.net = net
        self.coll = MCTSCollector(net, puct)
        self.log = MCTSLog(self.coll)
        self.game = GameState(self.log)
        for i in range(num_players):
            self.game.add_player(MCTSTrainerStrategy(self.coll))
        #
        self.start_phase = None
        self.reset_game()

    def reset_game(self):
        self.game._init_random()
        # print(self.coll.N1)
        self.coll.clear()
        self.start_phase = self.game.export_phaseinfo()
        self.played_games.add(self.start_phase)

    def sim_game_full(self, sims=5):
        for _ in range(sims):
            if len(self.examples) >= self.N:
                return True
            self.game._init_phaseinfo(self.start_phase)
            self.log.memories.clear()
            self.game.start_loop()
            self.examples += self.log.memories
        return False  # rerun to collect more examples
