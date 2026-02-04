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
from enum import IntEnum

# https://mcts.ai/code/python.html

# https://github.com/suragnair/alpha-zero-general
# f1c98a9b37e4b3bdddd4595966a7398829795817


class StateValuation(IntEnum):
    UNSEEN = 0
    FANOUT = 1
    PREDICTED = 2


class BatchedMCTSCollector:
    def __init__(self, net, puct, epsilon=1e-8, bound=30, batch_size=128):
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
        self.depth = dict()  # [s] -> phase_count (basically)
        self.repeats = CounterDict()  # [s] -> number of times we called search(s)
        self.batch_size = batch_size
        self.C = DupeFailDict()  # [s] -> combos
        self.vals = dict()  # [s] -> v (filled backwards?)
        #
        self.shortcut = None
        self.last_queried_phase = None

    def __len__(self):
        return len(self.N0)

    def reset_sim(self):
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
        self.repeats.clear()
        self.C.clear()
        self.vals.clear()
        #
        self.reset_sim()

    def search(self, phase, combos):
        cur_s = self.phases[phase]
        self.last_queried_phase = phase
        if self.shortcut is not None:
            # print(f"taking a shortcut at {cur_s}")
            a = self.shortcut
            self.shortcut = None
        else:
            # print(f"doing a search at {cur_s}")
            a = self._search(cur_s, phase, combos)
        return a

    def _search(self, s, phase, combos):
        # print("trying to _search at", s, phase)
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
            return -1
        #
        if s not in self.P:
            # print(f"getting probs for {s}", phase, phase.game_endvalue)
            # normalized probs
            p_hat, v_hat = self.net.predict(MCTSSample.eval(phase))
            p_hat = p_hat + (1e-8 * self.C[s])
            self.P[s] = normalize_probs(p_hat * self.C[s])
            self.vals[s] = v_hat
            self.N0[s] = 0
            self.N1[s] = np.zeros(128, dtype=np.int32)
        return random.choices(range(128), weights=self.P[s], k=1)[0]

    def sample_actions(self, s, phase, sims):
        if s not in self.P:
            self._search(s, phase, [])
            assert s in self.P, f"probs missing for ({s}) {phase}"
        return random.choices(range(128), weights=self.P[s], k=sims)

    def _search_ucb(self, s, phase, combos):
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

    def draw_edge(self, prev_s, prev_a, cur_s):
        self._connect(prev_s, prev_a, cur_s)

    def _connect(self, prev_s, prev_a, cur_s):
        if prev_s is None:
            self.depth[cur_s] = 0
            return
        # print("adding edge", (prev_s, prev_a), "->", cur_s)
        self.repeats[cur_s] = self.repeats.get(cur_s, 0) + 1
        self.f_edges[(prev_s, prev_a)] = cur_s
        self.b_edges[cur_s] = (prev_s, prev_a)  # ouch
        self.depth[cur_s] = self.depth.get(prev_s, 0) + 1

    def calc_policy(self, s):
        den = self.N0[s] + self.epsilon
        num_actions = len(self.C[s])
        arr = np.zeros(num_actions, dtype=np.float32)
        arr = (self.N1[s] * self.C[s]) / den
        arr = normalize_probs(arr)
        # print(f"policy for {s} is", arr)
        return arr

    def get_example(self, s):
        phase = self.phases.inverse(s)
        example = MCTSSample(phase, self.calc_policy(s), self.E[s], self.C[s])
        return example

    def evaluate_futures(self, cur_s, futures):
        s_estims = []
        estim_ct = 0
        bk_estim = DupeFailDict()
        uniqs = set()
        for i, (a, next_phase) in enumerate(futures):
            if next_phase.game_endvalue == 0:
                bk_estim[i] = estim_ct
                s_estims.append(MCTSSample.eval(next_phase))
                estim_ct += 1
                uniqs.add(next_phase)

        # print(f"evaluating {len(futures)} futures for {cur_s}")
        # print(f"unique futures: {len(uniqs)}")
        # TODO: call batch_predict only for unique futures?
        if len(s_estims) > 0:
            batch_inds = list(range(0, len(s_estims), self.batch_size))
            v_hat = np.zeros(len(s_estims), dtype=np.float32)
            if batch_inds[-1] != len(s_estims):
                batch_inds.append(len(s_estims))
            for j in range(0, len(batch_inds) - 1):
                j_start = batch_inds[j]
                j_end = batch_inds[j + 1]
                sub = s_estims[j_start:j_end]
                p0, v0 = self.net.batch_predict(sub)
                v_hat[j_start:j_end] = v0

        # updates happen after we have values (or estimates) for all states
        s = cur_s
        for i, (a, next_phase) in enumerate(futures):
            if i in bk_estim:
                v = v_hat[bk_estim[i]]
            else:
                assert next_phase.game_endvalue != 0, "no estim"
                v = next_phase.game_endvalue
            if (s, a) in self.Q:
                q1 = self.N1[s][a] * self.Q[(s, a)] + v
                q2 = 1 + self.N1[s][a]
                self.Q[(s, a)] = q1 / q2
                self.N1[s][a] += 1
            else:
                self.Q[(s, a)] = v
                self.N1[s][a] = 1

            self.N0[s] += 1

        # now call ucb to pick action
        cur_phase = self.phases.inverse(cur_s)
        best_act = self._search_ucb(cur_s, cur_phase, None)
        return best_act


class BatchedMCTS:
    def __init__(self, net, puct=0.1, N=1000, batch_size=16, randomize=False):
        self.N = N
        self._examples = list()
        #
        self.randomize = randomize
        self.num_players = random.randint(2, 4)
        self.net = net
        self.coll = BatchedMCTSCollector(net=net, puct=puct, batch_size=batch_size)
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
        return self._examples

    def clear_examples(self):
        self._examples.clear()

    def _sim_game_step(self, cur_phase, action):
        self.coll.reset_sim()
        self.game._init_phaseinfo(cur_phase)
        self.coll.shortcut = action
        self.game._step()
        if action == -1:
            next_phase = self.coll.last_queried_phase
            self.coll.last_queried_phase = None
            # print("got a queried phase", self.coll.phases[next_phase])
        else:
            next_phase = self.game.export_phaseinfo()
        return next_phase

    def _sim_game_from(self, cur_phase, sims):
        if cur_phase.game_endvalue != 0:
            return None, None
        #
        self._sim_game_step(cur_phase, None)
        cur_s = self.coll.phases[cur_phase]
        actions = self.coll.sample_actions(cur_s, cur_phase, sims)
        futures = []
        #
        for i, a in enumerate(actions):
            next_phase = self._sim_game_step(cur_phase, a)
            futures.append((a, next_phase))

        self._examples.append(self.coll.get_example(cur_s))
        #
        best_act = self.coll.evaluate_futures(cur_s, futures)
        next_phase = self._sim_game_step(cur_phase, best_act)
        next_s = self.coll.phases[next_phase]
        # print("drawing edge to", next_s)
        self.coll.draw_edge(cur_s, best_act, next_s)
        # print(self.coll.f_edges)
        return next_phase, next_s

    def sim_game_full(self, sims=5):
        phase = self.start_phase
        while phase.game_endvalue == 0:
            phase, cur_s = self._sim_game_from(phase, sims)

        for exp in self._examples:
            exp.value = phase.game_endvalue
        return True
