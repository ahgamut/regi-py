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
    def __init__(self, net, puct, epsilon=1e-8, bound=120):
        self.bound = bound
        self.net = net
        self.puct = puct
        self.epsilon = epsilon
        self.phases = PhaseLoader()  # [PhaseInfo] -> s
        #
        self.f_edges = DupeFailDict()  # [(s, a)] -> s'
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
        self.vals = dict()  # [s] -> v (filled backwards?)
        #

    def __len__(self):
        return len(self.N0)

    def reset_sim(self):
        pass

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

    def calc_policy(self, s):
        den = self.N0[s] + self.epsilon
        arr = np.zeros(len(self.C[s]), dtype=np.float32)
        arr = (self.N1[s] * self.C[s]) / den
        if self.depth.get(s, 0) > self.bound:
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
            self.vals[s] = 1.0 - (enemy_hp_left(phase) / 360.0)
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

    def _search_ucb(self, s, phase):
        # ucb
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        actions = self.C[s].nonzero()[0]
        for a0 in actions:
            a = int(a0)
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

        return best_act

    def _connect(self, prev_s, prev_a, cur_s):
        self.repeats[cur_s] = self.repeats.get(cur_s, 0) + 1
        if prev_s is None:
            self.depth[cur_s] = 0
            return
        # print("adding edge", (prev_s, prev_a), "->", cur_s)
        self.f_edges[(prev_s, prev_a)] = cur_s
        self.b_edges[cur_s] = (prev_s, prev_a)  # ouch
        self.depth[cur_s] = self.depth.get(cur_s, 0) + 1

    def update_backwards(self, next_s):
        v = self.vals[next_s]

        s0_stack = [next_s]
        while len(s0_stack) > 0:
            s0 = s0_stack[-1]
            for x in self.b_edges.get(s0, []):
                s, a = x
                if s is None:
                    continue
                if (s, a) in self.Q:
                    q1 = self.N1[s][a] * self.Q[(s, a)] + v
                    q2 = 1 + self.N1[s][a]
                    self.Q[(s, a)] = q1 / q2
                    self.N1[s][a] += 1
                else:
                    self.Q[(s, a)] = v
                    self.N1[s][a] = 1

                self.N0[s] += 1
                s0_stack.append(s)

            s0_stack.pop(0)

    def sim_temp_games(self, root_phase, max_sims):
        log = DummyLog()
        tmp = GameState(log)
        exp_strat = MCTSExplorerStrategy(root_phase)
        for i in range(root_phase.num_players):
            tmp.add_player(exp_strat)

        tmp._init_phaseinfo(root_phase)
        tmp.start_loop()
        root_combos = exp_strat.root_combos
        exp_strat.is_recording = False

        for i in range(len(root_combos)):
            exp_strat.shortcut = i
            tmp._init_phaseinfo(root_phase)
            tmp.start_loop()
            if exp_strat.next_phases[i] is None:
                exp_strat.next_phases[i] = tmp.export_phaseinfo()

        next_phases = exp_strat.next_phases
        root_s = self.phases[root_phase]
        for i, next_phase in enumerate(next_phases):
            assert next_phase is not None
            root_a = root_combos[i].bitwise
            next_s = self.phases[next_phase]
            # print(f"drawing {root_s, root_a} -> {next_s}")
            assert root_s != next_s
            self._connect(root_s, root_a, next_s)

        self._search(root_s, root_phase, root_combos)
        self._run_temp_sims(root_s, root_phase, root_combos, max_sims)

    def _run_temp_sims(self, root_s, root_phase, root_combos, max_sims):
        log = DummyLog()
        tmp = GameState(log)
        exp_strat = RandomStrategy()
        for i in range(root_phase.num_players):
            tmp.add_player(exp_strat)
        #
        actions = self.C[root_s].nonzero()[0]
        for _ in range(max_sims):
            best_a = self._search_ucb(root_s, root_phase)
            next_s = self.f_edges[(root_s, best_a)]
            next_phase = self.phases.inverse(next_s)
            tmp._init_phaseinfo(next_phase)
            tmp.start_loop()
            vstart = enemy_hp_left(next_phase)
            vend = enemy_hp_left(tmp.export_phaseinfo())
            val = (vstart - vend) / 360
            self.vals[next_s] = val
            self.update_backwards(next_s)

    def get_example(self, s):
        phase = self.phases.inverse(s)
        example = MCTSSample(phase, self.calc_policy(s), self.E[s], self.C[s])
        return example


class MCTS:
    def __init__(self, net, puct=1.25, N=1000, batch_size=16, randomize=False):
        self.N = N
        self._examples = list()
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
        return self._examples

    def clear_examples(self):
        self._examples.clear()

    def _sim_game_from(self, phase, sims):
        self.coll.sim_temp_games(phase, sims)
        cur_s = self.coll.phases[phase]
        # print("simmed", cur_s)
        self._examples.append(self.coll.get_example(cur_s))
        # get best action according to ucb
        best_a = self.coll._search_ucb(cur_s, phase)
        next_s = self.coll.f_edges[(cur_s, best_a)]
        # prob = self.coll.P[cur_s]
        # pol = self.coll.N1[cur_s] / self.coll.N0[cur_s]
        # print(f"{cur_s, best_a} -> {next_s} prob={prob[best_a]}, pol={pol[best_a]}")
        next_phase = self.coll.phases.inverse(next_s)
        return next_phase

    def _collect_examples(self, sims, end_phase):
        diffe = 1 - (enemy_hp_left(end_phase) / 360.0)
        for exp in self._examples:
            exp.value = diffe

    def sim_game_full(self, sims=5):
        phase = self._sim_game_from(self.start_phase, sims)
        while phase.game_endvalue == 0:
            phase = self._sim_game_from(phase, sims)

        self._collect_examples(sims, phase)
        return True
