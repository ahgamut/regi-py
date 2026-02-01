from regi_py.core import *
from regi_py.rl.mcts_utils import *
from regi_py.logging import DummyLog
from regi_py.rl.mcts import MCTSTrainerStrategy
from regi_py.rl.mcts import MCTSTesterStrategy

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


def _normalize_probs(arr):
    t = np.sum(arr)
    if t != 0:
        arr /= t
    else:
        # this is probably for terminal cases
        arr[0] = 1.0
    return arr


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
        self.N1 = CounterDict()  # [(s,a)] -> int
        self.P = dict()  # [s] -> probability vector
        self.E = DupeFailDict()  # [s] -> ??? (only for endgame?)
        #
        self.depth = dict()  # [s] -> phase_count (basically)
        self.partials = dict()  # [s] -> enum indicating when net was called or not
        self.repeats = CounterDict()  # [s] -> number of times we called search(s)
        self.batch_size = batch_size
        self.predicts = list()
        self.fanout_count = 0
        self.C = DupeFailDict()  # [s] -> combos
        self.vals = dict()  # [s] -> v (filled backwards?)
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
        self.partials.clear()
        self.repeats.clear()
        self.fanout_count = 0
        self.predicts.clear()
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

    def _obtain_fanout_probs(self, s):
        self.P[s] = np.ones((128,), dtype=np.float32) * self.C[s]
        self.P[s] = _normalize_probs(self.P[s])
        self.N0[s] = 0
        self.partials[s] = StateValuation.FANOUT
        self.fanout_count += 1

    def _obtain_network_probs(self, forced):
        if forced or self.fanout_count >= self.batch_size:
            batch = []
            for s, val in self.partials.items():
                if val == StateValuation.FANOUT:
                    batch.append(s)
                if len(batch) > self.batch_size:
                    break
            if len(batch) == 0:
                return
            if len(batch) > self.batch_size:
                subbatch = random.sample(batch, self.batch_size)
            else:
                subbatch = batch
            #
            self._call_network_with_batch(subbatch)

    def _call_network_with_batch(self, indbatch):
        batch = []
        for s in indbatch:
            spl = self.phases.inverse(s)
            batch.append(MCTSSample.eval(spl))
        p_hat, v_hat = self.net.batch_predict(batch)
        for i, smp in enumerate(batch):
            s = self.phases[smp.phase]
            self.P[s] = _normalize_probs(p_hat[i, :] * self.C[s])
            self.vals[s] = v_hat[i]
            self.repeats[s] = 0
            self.partials[s] = StateValuation.PREDICTED
            if np.sum(self.C[s]) > 1:
                self.predicts.append(s)
        #
        for i, smp in enumerate(batch):
            s = self.phases[smp.phase]
            self.update_backwards(s)
        self.fanout_count -= len(batch)

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
            if s not in self.P:
                self._obtain_fanout_probs(s)
            # print("ending at", s, "with", phase.game_endvalue)
            self.N0[s] = 1
            self.vals[s] = phase.game_endvalue
            self.partials[s] = StateValuation.PREDICTED
            self.fanout_count -= 1
            self.update_backwards(s)
            return -1
        if s not in self.P:
            # print("getting fanout probs for", s)
            self._obtain_fanout_probs(s)
            return random.choices(range(128), weights=self.P[s], k=1)[0]

        # self._obtain_network_probs(forced=False)

        if self.partials[s] == StateValuation.FANOUT:
            return self._search_fanout(s, phase, combos)

        return self._search_ucb(s, phase, combos)

    def _search_fanout(self, s, phase, combos):
        best_act = -1

        # we'd like to explore as much as possible
        for a in range(len(combos)):
            prob_a = self.P[s][a]
            if prob_a == 0:
                continue
            if (s, a) not in self.Q:
                best_act = a
                break

        if best_act != -1:
            best_act = random.choices(range(128), weights=self.P[s], k=1)[0]

        # print(f"exploring {best_act} from {s}")
        return best_act

    def _search_ucb(self, s, phase, combos):
        assert self.partials[s] == StateValuation.PREDICTED, "ucb only with predictions"

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
                # update Q only if we already have predictions
                if self.partials[s] == StateValuation.PREDICTED:
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

    def calc_policy(self, s):
        den = self.N0[s] + self.epsilon
        num_actions = len(self.C[s])
        arr = np.zeros(num_actions, dtype=np.float32)
        if len(arr) > 0:
            for a in range(num_actions):
                arr[a] = (self.N1[(s, a)] * self.C[s][a]) / den
            if self.depth[s] > self.bound:
                best = np.argmax(arr)
                arr *= 0
                arr[best] = 1.0

        arr = _normalize_probs(arr)
        # print(f"policy for {s} is", arr)
        return arr

    def fillout_phase_tree(self, cur_s):
        res = []

        # every descendant of s should have seen the net
        def fwd_check(s):
            oth = []
            if len(res) >= self.batch_size:
                return
            for a in range(128):
                for next_s in self.f_edges.get((s, a), []):
                    if self.partials[next_s] != StateValuation.PREDICTED:
                        res.append(next_s)
                    oth.append(next_s)
            for next_s in oth:
                fwd_check(next_s)

        # every ancestor of s should have seen the net
        def bkd_check(s):
            oth = []
            if len(res) >= self.batch_size:
                return
            for prev_s, prev_a in self.b_edges.get(s, []):
                if prev_s is None:
                    continue
                if self.partials[prev_s] != StateValuation.PREDICTED:
                    res.append(prev_s)
                oth.append(prev_s)
            for prev_s in oth:
                bkd_check(prev_s)

        fwd_check(cur_s)
        bkd_check(cur_s)
        if len(res) > 0:
            self._call_network_with_batch(res)

    def get_explorable_phase(self, max_sims):
        if len(self.predicts) == 0:
            return None

        i = random.randint(0, len(self.predicts) - 1)
        s = self.predicts[i]

        while self.repeats[s] >= max_sims or self.E[s] != 0:
            self.predicts.pop(i)
            if len(self.predicts) == 0:
                return None
            i = random.randint(0, len(self.predicts) - 1)
            s = self.predicts[i]

        self.predicts.pop(i)
        self.fillout_phase_tree(s)
        phase = self.phases.inverse(s)
        return phase

    def get_end_phases(self):
        res = []
        for s, val in self.E.items():
            if val != 0:
                phase = self.phases.inverse(s)
                res.append(phase)
        return res

    def rewardize(self, end_phase, max_sims):
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


class BatchedMCTSLog(DummyLog):
    def __init__(self, coll):
        super().__init__()
        self.coll = coll

    ####
    def endgame(self, reason, game):
        end_phase = game.export_phaseinfo()
        self.coll.search(end_phase, [])
        self.coll.reset_sim()


class BatchedMCTS:
    def __init__(self, net, batch_size, puct=0.1, N=1000):
        self.N = N
        self._examples = dict()
        self.older_samples = list()
        self.ended_games = set()
        #
        self.num_players = random.randint(2, 4)
        self.net = net
        self.coll = BatchedMCTSCollector(net=net, puct=puct, batch_size=batch_size)
        self.log = BatchedMCTSLog(self.coll)
        #
        self.start_phase = None
        self.reset_game()

    def get_examples(self):
        return self.older_samples + list(self._examples.values())

    def clear_examples(self):
        self.older_samples.clear()
        self._examples.clear()

    def count_examples(self):
        a = len(self.older_samples)
        b = len(self._examples)
        return a + b

    def reset_game(self):
        # print("game reset")
        self.coll.clear()
        self.older_samples += list(self._examples.values())
        self._examples.clear()
        self.ended_games.clear()
        self.num_players = random.randint(2, 4)
        self.game = GameState(self.log)
        for i in range(self.num_players):
            self.game.add_player(MCTSTrainerStrategy(self.coll))
        self.game._init_random()
        self.start_phase = self.game.export_phaseinfo()

    def _sim_game_from(self, phase, sims):
        cur_s = self.coll.phases[phase]
        self.coll.repeats[cur_s] = 0
        for _ in range(sims):
            self.coll.reset_sim()
            self.game._init_phaseinfo(phase)
            self.coll.repeats[cur_s] = self.coll.repeats[cur_s] + 1
            self.game.start_loop()
        self._collect_examples(sims)

    def _collect_examples(self, sims):
        end_list = self.coll.get_end_phases()
        for end in end_list:
            if end in self.ended_games:
                continue
            r1 = self.coll.rewardize(end, sims)
            for s, exp in r1.items():
                if self.coll.repeats[s] < sims:
                    continue
                v0 = self._examples.get(s)
                if v0 is None:
                    self._examples[s] = exp
                elif exp.value == v0.value:
                    # we already have this, so update policy only
                    self._examples[s].policy = exp.policy
                else:  # different result
                    self._examples[-s] = exp
                if self.count_examples() > self.N:
                    return
            self.ended_games.add(end)

    def show_new_samples(self):
        for i, (s, x) in enumerate(self._examples.items()):
            s = self.coll.phases[x.phase]
            nzp = len(x.policy.nonzero()[0])
            print(
                f"{i} (phase #{s}) repeats={self.coll.repeats[s]}, value={x.value}, probs={nzp}"
            )

    def sim_game_root(self, sims):
        self._sim_game_from(self.start_phase, sims)
        root_s = self.coll.phases[self.start_phase]
        self.coll.fillout_phase_tree(root_s)

    def sim_game_full(self, sims=5):
        self.sim_game_root(sims)
        phase = self.start_phase
        while phase is not None:
            if self.count_examples() > self.N:
                break
            self._sim_game_from(phase, sims)
            phase = self.coll.get_explorable_phase(sims)
            if phase is None:
                self.coll._obtain_network_probs(forced=True)
                phase = self.coll.get_explorable_phase(sims)

        # self.show_new_samples()
        return self.count_examples() > self.N
