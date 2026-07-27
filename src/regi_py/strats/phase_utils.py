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
    return random.random() <= 0.7


def defend_throwing(ind, game, combos, score_only=False):
    if len(combos) < 4:
        return False
    sel_blk = combos[ind].base_defense
    num_discards = len(combos[ind].parts)
    lower_poss = 0
    for c in combos:
        c_blk = c.base_defense
        c_dsc = len(c.parts)
        if c_dsc < num_discards and c_blk <= sel_blk:
            lower_poss += 0.5
    lower_prob = min(0.99, lower_poss)
    if score_only:
        return lower_prob
    return random.random() <= lower_prob


def get_nonbad_attacks(game, combos):
    res = []
    if len(combos) < 4:
        return combos
    for ind, c in enumerate(combos):
        if c.bitwise != 0:
            res.append(c)
            continue
        if random.random() > 0.6:
            res.append(c)

    return res


def get_preserve_attacks(player, combos, game):
    if len(combos) < 3:
        return combos
    e = game.enemy_pile[0]
    cur_block = game.get_current_block(e)
    ac_map = {str(x): x for x in player.cards}
    all_cards = set(x for x in player.cards)

    good_combos = []
    for ind, x in enumerate(combos):
        dmg = game.get_combo_damage(e, x)
        cset = set(c for c in x.parts)
        remain = list(c for c in (all_cards - cset))
        new_blk = game.get_combo_block(e, x)
        remain_blk = sum(c.strength for c in remain)
        # print(x, "attacks for", dmg, "blocks for", new_blk)
        # print("we already block for", cur_block)
        # print("remaining cards can block at most", remain_blk)

        if remain_blk >= e.strength - (cur_block + new_blk):
            good_combos.append(x)

    if len(good_combos) == 0:
        return combos

    good_combos = list(
        sorted(
            good_combos,
            reverse=True,
            key=lambda x: game.get_combo_damage(e, x),
        )
    )
    # print(len(combos) - len(good_indices), "combos were removed")
    return good_combos


def get_nonbad_defends(game, combos):
    if len(combos) < 4:
        return combos
    res = []
    N = len(combos)
    for ind in range(N):
        if not defend_throwing(ind, game, combos):
            res.append(combos[ind])
    return res


class PhaseRecorderStrategy(BaseStrategy):
    __strat_name__ = "phase-recorder"

    def __init__(self, root_phase):
        super(PhaseRecorderStrategy, self).__init__()
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

    def getRedirectIndex(self, player, game):
        offset = random.randint(1, game.num_players - 1)
        return (game.active_player + offset) % game.num_players

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
            subind = self.shortcut
            self.shortcut = None
            bts = self.root_combos[subind].bitwise
            for i, c in enumerate(combos):
                if c.bitwise == bts:
                    ind = i
                    break
            self.update_prev(subind)
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


def indexify(move, combos):
    for i, c in enumerate(combos):
        if c.bitwise == move.bitwise:
            return i
    raise RuntimeError("unable to index move")
    return 0


class _ExpansionStrategy(BaseStrategy):
    """Forcing strategy for get_expansion_at.

    It is consulted only at decision points (an attack/defense where a combo is
    chosen).  At the root decision it records the offered combos and plays the
    one whose ``bitwise`` matches ``force_bitwise`` (or index 0 while just
    discovering the root combos).  At the *first* decision after that -- the
    child node -- it snapshots the phase into ``captured`` so the driver knows to
    stop stepping.  Phases that resolve without a decision (e.g. an auto-blocked
    counterattack) are stepped over, matching "next decision node" semantics.
    """

    __strat_name__ = "expansion"

    def __init__(self):
        super(_ExpansionStrategy, self).__init__()
        self.root_offered = None
        self.force_bitwise = None
        self.at_root = True
        self.captured = None

    def arm(self, force_bitwise):
        self.force_bitwise = force_bitwise
        self.at_root = True
        self.captured = None

    def setup(self, player, game):
        return 0

    def getRedirectIndex(self, player, game):
        offset = random.randint(1, game.num_players - 1)
        return (game.active_player + offset) % game.num_players

    def _choose(self, combos, game):
        if self.at_root:
            self.root_offered = list(combos)
            self.at_root = False
            if self.force_bitwise is not None:
                for i, c in enumerate(combos):
                    if c.bitwise == self.force_bitwise:
                        return i
            return 0
        # first decision reached after the root combo == the child phase
        if self.captured is None:
            self.captured = game.export_phaseinfo()
        return 0

    def getAttackIndex(self, combos, player, yield_allowed, game):
        return self._choose(combos, game)

    def getDefenseIndex(self, combos, player, damage, game):
        return self._choose(combos, game)


def get_expansion_at(root_phase, trim=False):
    """Enumerate the children of ``root_phase``: the combos legally playable at
    the root and, for each, the phase at the next decision node reached after
    playing it (or the terminal phase if the game ends).

    Each child is expanded by re-seating the root phase, forcing that combo, and
    stepping only until the next decision -- not replaying a whole game.
    """
    tmp = GameState(DummyLog())
    tmp.record_history = False  # throwaway: history is never read here
    strat = _ExpansionStrategy()
    for _ in range(root_phase.num_players):
        tmp.add_player(strat)

    def expand(force_bitwise):
        strat.arm(force_bitwise)
        tmp._init_phaseinfo(root_phase)
        if not tmp.is_runnable:
            return None
        tmp._step()  # the root decision (plays the forced combo)
        # advance to the next decision node (or the end of the game)
        while tmp.is_runnable and strat.captured is None:
            tmp._step()
        if strat.captured is not None:
            return strat.captured
        return tmp.export_phaseinfo()

    # discover the combos offered at the root
    expand(None)
    root_combos = strat.root_offered
    if root_combos is None:
        return [], []

    if trim:
        if root_phase.phase_attacking:
            root_combos = get_nonbad_attacks(None, root_combos)
        else:
            root_combos = get_nonbad_defends(None, root_combos)

    next_phases = [expand(combo.bitwise) for combo in root_combos]
    assert len(next_phases) == len(root_combos)
    return next_phases, root_combos


class QuickLog(DummyLog):
    def __init__(self):
        super().__init__()
        self.reason = None

    def endgame(self, reason, game):
        self.reason = reason


def quick_game_sim(root_phase, strat_klass):
    log = QuickLog()
    tmp = GameState(log)
    # throwaway rollout: only the final state is read, so skip the per-phase
    # history snapshot allocation
    tmp.record_history = False
    exp_strat = strat_klass()
    for i in range(root_phase.num_players):
        tmp.add_player(exp_strat)
    tmp._init_phaseinfo(root_phase)
    tmp.start_loop()
    return tmp, log.reason


def quick_game_value(root_phase, strat_klass, relative_diff=False):
    game, reason = quick_game_sim(root_phase, strat_klass)
    # give bad values if due to move failure
    end_phase = game.export_phaseinfo()
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
