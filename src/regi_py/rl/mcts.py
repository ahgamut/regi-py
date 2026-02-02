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
        self.N1 = dict()  # [s] -> array of ints
        self.P = dict()  # [s] -> probability vector
        self.E = DupeFailDict()  # [s] -> ??? (only for endgame?)
        #
        self.repeats = dict()  # [s] -> number of times we called search(s)
        self.depth = dict()  # [s] -> phase_count (basically)
        self.C = DupeFailDict()  # [s] -> combos
        self.vals = DupeFailDict()  # [s] -> v (filled backwards?)
        #
        self.prev_s = None
        self.prev_a = None
        self.shortcut = None

    def __len__(self):
        return len(self.N0)

    def reset_sim(self):
        self.prev_s = None
        self.prev_a = None
        self.shortcut = None

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
        if self.shortcut is not None:
            a = self.shortcut
            self.shortcut = None
        else:
            a = self._search(cur_s, phase, combos)
        self.prev_s = cur_s
        self.prev_a = a
        # print(f"picked ({self.prev_s}, {self.prev_a})")
        return a

    def calc_policy(self, s):
        den = self.N0[s] + self.epsilon
        arr = np.zeros(len(self.C[s]), dtype=np.float32)
        arr = (self.N1[s] * self.C[s]) / den
        if self.depth[s] > self.bound:
            best = np.argmax(arr)
            arr *= 0
            arr[best] = 1.0
        # print(f"policy for {s} is", arr)
        return normalize_probs(arr)

    def _search(self, s, phase, combos):
        if s not in self.C:
            cbr = np.zeros(128)
            for x in combos:
                cbr[x.bitwise] = 1
            self.C[s] = cbr
            self.N1[s] = np.zeros(128, dtype=np.float32)
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
            p_hat, v_hat = self.net.predict(MCTSSample.eval(phase))
            self.P[s] = normalize_probs(p_hat * self.C[s])
            self.vals[s] = v_hat
            self.N0[s] = 0
            self.update_backwards(s)
            return random.choices(range(128), weights=self.P[s], k=1)[0]

        # ucb
        cur_best = -float("inf")
        best_act = 0

        # pick the action with the highest upper confidence bound
        for a in range(len(self.C[s])):
            prob_a = self.P[s][a]
            if (s, a) in self.Q:
                u1 = self.Q[(s, a)]
                u2 = self.puct * prob_a * np.sqrt(self.N0[s]) / (1 + self.N1[s][a])
                u = u1 + u2
            else:  # Q = 0 ?
                u = self.puct * prob_a * np.sqrt(self.N0[s] + self.epsilon)

            if u > cur_best:
                cur_best = u
                best_act = a

        # print(f"best_act from {s} is {best_act} (u={cur_best})")
        return best_act

    def _connect(self, prev_s, prev_a, cur_s):
        self.repeats[cur_s] = self.repeats.get(cur_s, 0) + 1
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
                    q1 = self.N1[s][a] * self.Q[(s, a)] + v
                    q2 = 1 + self.N1[s][a]
                    self.Q[(s, a)] = q1 / q2
                    self.N1[s][a] += 1
                else:
                    self.Q[(s, a)] = v
                    self.N1[s][a] = 1

                self.N0[s] += 1
                set_qn(s)

        set_qn(next_s)

    def get_explorable_phase(self, max_sims):
        explore_s = None
        for s, sim in self.repeats.items():
            if sim >= max_sims or self.E.get(s, -1) != 0:
                continue
            explore_s = s
            break

        if explore_s is None:
            return None
        return self.phases.inverse(explore_s)

    def get_end_phases(self):
        res = []
        for s, val in self.E.items():
            if val != 0:
                phase = self.phases.inverse(s)
                res.append(phase)
        return res

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


class MCTS:
    def __init__(self, net, puct=0.1, N=1000, batch_size=16, randomize=False):
        self.N = N
        self._examples = dict()
        #
        self.net = net
        self.randomize = randomize
        self.coll = MCTSCollector(net, puct)
        self.log = MCTSLog(self.coll)
        #
        self.start_phase = None
        self.reset_game()

    def reset_game(self):
        self.num_players = random.randint(2, 4)
        self.game = GameState(self.log)
        for i in range(self.num_players):
            self.game.add_player(MCTSTrainerStrategy(self.coll))
        if self.randomize:
            self.game._init_random()
        else:
            self.game.initialize()
        self.coll.clear()
        self.start_phase = self.game.export_phaseinfo()

    def get_examples(self):
        return list(self._examples.values())

    def clear_examples(self):
        self._examples.clear()

    def _sim_game_from(self, phase, sims):
        cur_s = self.coll.phases[phase]
        for _ in range(sims):
            self.coll.reset_sim()
            self.game._init_phaseinfo(phase)
            self.game.start_loop()

        # get best action according to ucb
        best_a = self.coll.search(phase, None)
        self.coll.reset_sim()
        # set as shortcut and run sim once so edge exists
        self.coll.shortcut = best_a
        self.game._init_phaseinfo(phase)
        self.game.start_loop()

        next_s = int(self.coll.f_edges[(cur_s, best_a)][0])
        if next_s not in self.coll.phases._inverse:
            print(self.coll.phases._inverse)
        phase = self.coll.phases.inverse(next_s)
        return phase

    def _collect_examples(self, sims, end_phase):
        r1 = self.coll.rewardize(end_phase)
        for s, exp in r1.items():
            self._examples[s] = exp

    def sim_game_full(self, sims=5):
        phase = self.start_phase
        cur_s = self.coll.phases[phase]
        self._sim_game_from(phase, sims)
        while self.coll.E[cur_s] == 0:
            phase = self._sim_game_from(phase, sims)
            cur_s = self.coll.phases[phase]

        self._collect_examples(sims, phase)
        return True
