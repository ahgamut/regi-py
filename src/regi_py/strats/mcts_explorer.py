from regi_py.core import BaseStrategy
from regi_py.strats.phase_utils import *
import random


class MCTSExplorerStrategy(BaseStrategy):
    __strat_name__ = "mcts-explorer"

    def __init__(self, root_phase):
        super(MCTSExplorerStrategy, self).__init__()
        self.root_phase = root_phase
        self.reroll()

    def reroll(self):
        self.shortcut = None
        self.root_combos = None
        self.next_phases = None
        #
        self.prev_phase = None
        self.prev_a = None
        self.is_recording = True

    def setup(self, player, game):
        return 0

    def mark_combo(self, phase):
        self.next_phases[self.prev_a] = phase

    def update_prev(self, a):
        self.prev_a = a

    def process_phase(self, phase, combos):
        if str(phase) == str(self.root_phase):
            if self.is_recording:
                self.root_combos = combos
                self.next_phases = [None] * len(combos)
            self.prev_phase = phase
        elif str(self.prev_phase) == str(self.root_phase):
            self.mark_combo(phase)
            self.prev_phase = phase
        if self.shortcut is not None:
            ind = self.shortcut
            self.shortcut = None
        else:
            ind = random.choice(range(len(combos)))
        self.update_prev(ind)
        return ind

    def getAttackIndex(self, combos, player, yield_allowed, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        ind = self.process_phase(game.export_phaseinfo(), combos)
        return ind


def get_expansion_at(root_phase, trim=False):
    log = DummyLog()
    tmp = GameState(log)
    exp_strat = MCTSExplorerStrategy(root_phase)
    for i in range(root_phase.num_players):
        tmp.add_player(exp_strat)

    tmp._init_phaseinfo(root_phase)
    tmp.start_loop()
    root_combos = exp_strat.root_combos
    exp_strat.is_recording = False

    if trim:
        if root_phase.phase_attacking:
            root_combos = get_nicer_attacks(None, root_combos)
        else:
            root_combos = get_nicer_defends(None, root_combos)
        exp_strat.root_combos = root_combos
        exp_strat.next_phases = [None] * len(root_combos)

    for i in range(len(root_combos)):
        exp_strat.shortcut = i
        tmp._init_phaseinfo(root_phase)
        tmp.start_loop()
        if exp_strat.next_phases[i] is None:
            exp_strat.next_phases[i] = tmp.export_phaseinfo()

    next_phases = exp_strat.next_phases
    assert len(next_phases) == len(root_combos)
    for x in next_phases:
        assert x is not None

    return next_phases, root_combos
