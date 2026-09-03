"""Generic, single-process trainer plumbing shared by both NN pipelines.

Everything here is reusable, mp-free: the self-play / brute / team / eval GAME
RUNNERS (``run_single_game`` / ``run_brute_game`` / ``run_team_game`` /
``test_model`` / ``improved_gameplay``), which are paradigm-AGNOSTIC -- each takes a
``paradigm`` (see ``trainer_loop.Paradigm``) bundling the AZ-vs-ADZ node / strategy /
recorder / ``infos_fn`` classes, so one body serves both; the optimization step
(``run_epoch``); the AZ-specific data recorders + ``infos_from_game`` (the ADZ ones
live in ``adz_training``); and small helpers (``EndGameLog``, ``total_enemy_hp``,
``get_split_optimizer``, ``drain``, ``sample_teammate``, the recorder mixins).
"""
import random
import sys
import time

#
import torch
import numpy as np

from regi_py import GameState, DummyLog, seed
from regi_py.core import MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS
from regi_py.combomap import cell_of_bitwise
from regi_py.strats import RandomStrategy, BruteSamplingStrategy, STRATEGY_LIST
from regi_py.rl.az.explorer import AZExplorerStrategy, AlphaZeroNode, AZNodeInfo
from regi_py.rl.utils import enemy_hp_left
from regi_py.rl.value_fns import assign_values, phase_snapshot


# brute teammates search at the full default depth
TEAM_BRUTE_ITERS = 128


def sample_teammate():
    """A random non-NN strategy instance from ``regi_py.strats`` (``STRATEGY_LIST``)
    to fill a team game's non-NN seats. All zero-arg except ``BruteSamplingStrategy``."""
    cls = random.choice(STRATEGY_LIST)
    if cls is BruteSamplingStrategy:
        return cls(iterations=TEAM_BRUTE_ITERS)
    return cls()


def total_enemy_hp(game):
    return sum(x.hp for x in game.enemy_pile)


class EndGameLog(DummyLog):
    def __init__(self):
        super().__init__()
        self.e0 = 0
        self.e1 = 0
        self.reason = None

    def startgame(self, game):
        self.e0 = total_enemy_hp(game)

    def endgame(self, reason, game):
        self.reason = reason
        self.e1 = total_enemy_hp(game)

    def diffe(self):
        return f"{self.e0-self.e1}({self.reason.value})"


def get_split_optimizer(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # keep all 1-D params out of weight decay: biases plus every norm-layer
        # affine weight/bias (GroupNorm now, was BatchNorm). Matching by ndim is
        # robust to layer renames, unlike the old "bn"/"batchnorm" name check.
        if param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)

    grps = [
        {"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0},
    ]

    optimizer = torch.optim.AdamW(grps, lr=5e-3)
    return optimizer


def drain(q, buf):
    while not q.empty():
        obj = q.get()
        buf.add(obj)  # COPIES the rows into the buffer's private ring tensors
        del obj  # frees the child's shared-memory (/dev/shm) segments right now


# max global grad-norm before an optimizer step; caps the damage from a single
# spiky batch. Named constant (tune per net if needed), like the value-fn scales.
_GRAD_CLIP_NORM = 10.0


def run_epoch(model, batch, optimizer):
    # a batch is a tuple of tensors in the model's ``TRAIN_FIELDS`` order (that is
    # how ``ShardBuffer.sample_batch`` stacks its ring tensors). Both paradigms
    # (card-space AZ nets and candidate-scoring ADZ nets) reach the net + loss the
    # same way: build ``data`` from the fields, forward, then let the net pull its
    # own target keys from ``data`` in ``calculate_loss(data, y_hat)``.
    data = {k: v.to(model.device) for k, v in zip(type(model).TRAIN_FIELDS, batch)}
    y_hat = model(data)
    loss, comps = model.calculate_loss(data, y_hat)
    if not torch.isfinite(loss):
        # a non-finite loss (transformer activation overflow -> inf/nan in the softmax)
        # would step nan into AdamW's moments and poison every later update; skip the
        # step so one spike can't kill the run
        print(
            f"WARN non-finite loss {loss.item()} comps="
            f"{tuple(c.item() for c in comps)}; skipping optimizer step",
            file=sys.stderr,
        )
        optimizer.zero_grad(set_to_none=True)
        return loss.item(), tuple(c.item() for c in comps)
    optimizer.zero_grad()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
    if not torch.isfinite(gnorm):
        # a finite loss can still backprop to nan/inf grads; clipping cannot rescue
        # those, so skip the step rather than poison AdamW's moments
        print(
            f"WARN non-finite grad norm {gnorm.item()}; skipping optimizer step",
            file=sys.stderr,
        )
        optimizer.zero_grad(set_to_none=True)
        return loss.item(), tuple(c.item() for c in comps)
    optimizer.step()
    # total plus (policy, value, keepy) components for per-head logging
    return loss.item(), tuple(c.item() for c in comps)


# self-play move selection: for the first SELFPLAY_TEMP_MOVES moves, sample the
# next move in proportion to child visit counts (AlphaZero temperature tau=1) so
# self-play games diversify and the replay buffer covers more distinct states;
# after that play greedily (tau -> 0) toward the strongest line. Greedy-from-move-1
# collapsed every game onto ~one trajectory.
SELFPLAY_TEMP_MOVES = 12


def _sample_selfplay_child(node, move_num):
    if move_num >= SELFPLAY_TEMP_MOVES:
        return node.best_child_node
    weights = [c.visits for c in node.children]
    if sum(weights) <= 0:
        return node.best_child_node
    return random.choices(node.children, weights=weights, k=1)[0]


def run_single_game(tid, i, net, num_bots, num_iterations, paradigm, value_fn):
    """Net self-play from a fresh ``initialize()`` (AZ or ADZ, per ``paradigm``):
    visit-count temperature for the opening then greedy. ``value_fn`` sets the value
    target per position. Returns the net's training tensors."""
    a = time.time()
    log = DummyLog()
    strat = RandomStrategy()
    game = GameState(log)
    for _ in range(num_bots):
        game.add_player(strat)
    game.initialize()
    start_phase = game.export_phaseinfo()
    #
    history = []
    actions = []  # played combo (card locations) per decision, for the value_fn
    node = paradigm.node_cls(start_phase, net=net, history=[], prior=1.0, trim=False)
    s0 = enemy_hp_left(node.root_phase)
    move_num = 0
    while node.root_phase.game_endvalue == 0:
        # root-only exploration noise: mix fresh Dirichlet into this move's
        # search root before searching (self-play only, not competitive play)
        node.add_dirichlet_noise()
        paradigm.simulate_fn(node, num_iterations)
        history.append(node.export())
        child = _sample_selfplay_child(node, move_num)
        # the combo played at this decision = the one that produced the chosen child
        actions.append([c.location for c in child.prev_combo.parts])
        child.parent = None
        node = child
        move_num += 1
    history.append(node.export())
    actions.append([])  # terminal export: no decision was made from it
    win = node.root_phase.game_endvalue == 1
    s1 = enemy_hp_left(node.root_phase)
    #
    dt = time.time() - a
    # dense trajectory: record k's decision phase is history[k].history[-1] (already a
    # by-value copy), positions 0..last map straight onto the snapshot
    snapshot = [info.history[-1] for info in history]
    assign_values(
        history, snapshot, list(range(len(history))), actions, win, s0, s1, value_fn
    )
    print(f"{tid},{i},p{len(history)},{s0},{s1},{dt:.4f}s,{win}", file=sys.stderr)
    return type(net).tensorify_training(history)


# Recording strategies stash, per decision, ONLY plain values (a history index +
# the played move) -- never a PhaseInfo reference, which would dangle when the
# game.history vector reallocates (segfault at tensorify). Windows are rebuilt from
# the stable post-game history in infos_from_game. Two mixins factor the shared
# attack/defense wrappers; each recorder supplies only the paradigm-specific record.
class _BruteRecordMixin:
    """For a ``BruteSamplingStrategy`` recorder: pick the brute move (same random
    fallback as the parent) then append it. Subclass implements ``_append(root_phase,
    combos, ind, game)`` -- the only per-paradigm bit (AZ vs ADZ record tuple)."""

    def _record_and_pick(self, combos, game):
        root_phase = game.export_phaseinfo()
        try:
            ind = self.get_best_move(root_phase, combos)
        except Exception as e:
            print("failed to process moves", e, file=sys.stderr)
            ind = random.randint(0, len(combos) - 1)
        self._append(root_phase, combos, ind, game)
        return ind

    def getAttackIndex(self, combos, player, yield_allowed, game):
        return self._record_and_pick(combos, game) if combos else -1

    def getDefenseIndex(self, combos, player, damage, game):
        return self._record_and_pick(combos, game) if combos else -1


class _TeamRecordMixin:
    """For an Explorer-strategy recorder on a team game's NN seat(s): play via the
    net (``super()``) then record the chosen decision. Because a strategy is consulted
    only on its own turns, this records exactly the NN-active decisions. Subclass
    implements ``_record(combos, ind, game)`` (skips a no-move ``ind``)."""

    def getAttackIndex(self, combos, player, yield_allowed, game):
        ind = super().getAttackIndex(combos, player, yield_allowed, game)
        self._record(combos, ind, game)
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        ind = super().getDefenseIndex(combos, player, damage, game)
        self._record(combos, ind, game)
        return ind


class RecordingBruteStrategy(_BruteRecordMixin, BruteSamplingStrategy):
    """Brute self-play recorder (AZ). One instance is shared by all players (Regicide
    is cooperative: every decision in a won game earns the win)."""

    def __init__(self, iterations=64):
        super().__init__(iterations=iterations)
        self.moves = []  # (history_index, combo_bitwise, [part_location, ...])

    def _append(self, root_phase, combos, ind, game):
        combo = combos[ind]
        idx = len(game.history) - 1  # decision phase is history[-1] (start_loop pushed it)
        self.moves.append((idx, combo.bitwise, [c.location for c in combo.parts]))


def infos_from_game(game, moves, win, s0, s1, net_cls, value_fn):
    """Build the ``AZNodeInfo`` training list AFTER a brute/team game finishes.

    Rebuilds each decision's window from the now-stable ``game.history`` (safe: the game
    is over so the vector no longer reallocates, and tensorify runs while ``game`` is
    still in scope). One-shot analogue of ``AlphaZeroNode.export()`` (N0 = 1):
    ``keepyness`` is 1 for kept hand cards / 0 for cards spent in the played combo;
    ``atk_probs`` is a one-hot at the played combo's ComboTable cell on attack phases
    (defense keeps it zero -- masked by ``attacking``). ``value_fn`` sets each value from
    the game snapshot; ``moves``' history indices are the snapshot positions.
    """
    hist = list(game.history)
    maxhist = net_cls.max_history
    infos = []
    positions = []
    actions = []
    for idx, bitwise, part_locs in moves:
        root_phase = hist[idx]
        window = AlphaZeroNode._trimmed_history(hist[:idx], root_phase, maxhist)
        #
        keepyness = np.zeros(MAX_CARDS_IN_GAME, dtype=np.float32)
        for card in root_phase.player_cards[root_phase.active_player]:
            keepyness[card.location] = 1.0
        for loc in part_locs:
            keepyness[loc] = 0.0
        #
        atk_probs = np.zeros((MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS), dtype=np.float32)
        if root_phase.phase_attacking:
            lp = cell_of_bitwise(bitwise)
            if lp is not None:
                atk_probs[lp] = 1.0
        #
        infos.append(
            AZNodeInfo(history=window, value=0.0, atk_probs=atk_probs, keepyness=keepyness)
        )
        positions.append(idx)
        actions.append(part_locs)
    assign_values(infos, phase_snapshot(hist), positions, actions, win, s0, s1, value_fn)
    return infos


def run_brute_game(tid, i, net_cls, num_bots, iterations, paradigm, value_fn):
    """Play one brute-sampling game from a random mid-game state; return training
    tensors only if it made large progress (else ``None``).

    ``init_random`` seeds a random mid-game position (partial deck, fewer enemies),
    so brute wins here yield short, diverse *late-game* trajectories -- the data net
    self-play (always from a fresh ``initialize()``) never produces. ``paradigm``
    supplies the AZ/ADZ brute recorder + ``infos_fn``.
    """
    a = time.time()
    log = EndGameLog()
    strat = paradigm.brute_recorder(iterations=iterations)
    game = GameState(log)
    for _ in range(num_bots):
        game.add_player(strat)
    game.init_random()
    s0 = enemy_hp_left(game.export_phaseinfo())
    if s0 < 170:
        return None
    game.start_loop()
    end_phase = game.export_phaseinfo()
    s1 = enemy_hp_left(end_phase)
    dt = time.time() - a
    win = end_phase.game_endvalue == 1
    if s0 - s1 < 170:  # only games that progress a lot are submitted as training data
        return None
    print(f"{tid},{i},b{len(strat.moves)},{s0},{s1},{dt:.4f}s,{win}", file=sys.stderr)
    # records built AFTER the game from the now-stable history; value_fn scores them
    infos = paradigm.infos_fn(
        game, strat.moves, win=win, s0=s0, s1=s1, net_cls=net_cls, value_fn=value_fn
    )
    if not infos:
        return None
    return net_cls.tensorify_training(infos)


class RecordingAZTeamStrategy(_TeamRecordMixin, AZExplorerStrategy):
    """``AZExplorerStrategy`` that records each of its own decisions brute-style
    (``(history_index, played_bitwise, [part_location, ...])``), rebuilt post-game by
    ``infos_from_game``. See the team-games design notes."""

    def __init__(self, net, iterations, moves):
        super().__init__(net, iterations=iterations, trim=False)
        self.moves = moves

    def _record(self, combos, ind, game):
        if ind is None or ind < 0 or not combos:
            return
        combo = combos[ind]
        idx = len(game.history) - 1
        self.moves.append((idx, combo.bitwise, [c.location for c in combo.parts]))


def run_team_game(tid, i, net, num_bots, iterations, paradigm, value_fn):
    """One full cooperative game: 1-2 net-guided-MCTS seats (``paradigm`` team recorder)
    + non-NN teammates from ``regi_py.strats``. Returns training tensors from the NN's
    decisions only (one-hot targets, value = outcome); every game submitted. See
    the team-games design notes."""
    a = time.time()
    log = EndGameLog()
    moves = []
    nn_strat = paradigm.team_recorder(net, iterations, moves)
    num_nn = random.randint(1, min(2, num_bots - 1))  # >= 1 non-NN teammate
    seats = [nn_strat] * num_nn + [sample_teammate() for _ in range(num_bots - num_nn)]
    random.shuffle(seats)
    game = GameState(log)
    for s in seats:
        game.add_player(s)
    game.initialize()
    s0 = enemy_hp_left(game.export_phaseinfo())
    game.start_loop()
    end_phase = game.export_phaseinfo()
    s1 = enemy_hp_left(end_phase)
    dt = time.time() - a
    win = end_phase.game_endvalue == 1
    print(
        f"{tid},{i},t{len(moves)}({num_nn}/{num_bots}),{s0},{s1},{dt:.4f}s,{win}",
        file=sys.stderr,
    )
    infos = paradigm.infos_fn(
        game, moves, win=win, s0=s0, s1=s1, net_cls=type(net), value_fn=value_fn
    )
    if not infos:
        return None
    return type(net).tensorify_training(infos)


def test_model(episode, model, num_simulations, paradigm):
    model.eval()
    log = EndGameLog()
    diffe = []
    for s in range(10):
        game = GameState(log)
        num_players = random.randint(2, 4)
        for i in range(num_players):
            game.add_player(paradigm.direct_strat(model))
        game.initialize()
        game.start_loop()
        diffe.append(log.diffe())
    print("test games:", diffe, file=sys.stderr)
    torch.save(model.state_dict(), f"./weights/model_{model.__mname__}_{episode}.pt")
    print("episode", episode, "saved model", file=sys.stderr)


def improved_gameplay(episode, new_model, old_model, num_simulations, paradigm, threshold=0.6):
    new_model.eval()
    old_model.eval()
    log1 = EndGameLog()
    log2 = EndGameLog()

    newer_better = 0
    # evaluate with net-guided MCTS (fixed 64 iters/move), not the search-free
    # Direct strategy, so the comparison reflects competitive play
    old_strat = paradigm.explorer_strat(old_model, iterations=64)
    new_strat = paradigm.explorer_strat(new_model, iterations=64)

    for s in range(num_simulations):
        game1 = GameState(log1)
        game2 = GameState(log2)
        #
        num_players = random.randint(2, 4)
        for i in range(num_players):
            game1.add_player(old_strat)
            game2.add_player(new_strat)
        # both games get the SAME starting RNG seed: identical deal, and each loop
        # replays from the same C++/Python rng stream, so the only difference is
        # which model drives the search (not luck of the draw or rollout order)
        game_seed = random.randint(0, 2**31 - 1)
        seed(game_seed)
        game1.initialize()
        game2.init_phaseinfo(game1.export_phaseinfo())
        #
        seed(game_seed)
        random.seed(game_seed)
        game1.start_loop()
        seed(game_seed)
        random.seed(game_seed)
        game2.start_loop()
        #
        diff1 = log1.e0 - log1.e1
        diff2 = log2.e0 - log2.e1
        #
        print(f"{s} old: {diff1}, new: {diff2}", file=sys.stderr)
        if diff2 > diff1 and diff2 != 0:
            newer_better += 1

    nb_ratio = newer_better / num_simulations
    print(f"{episode} newer better in {100*nb_ratio:.4f}% of games", file=sys.stderr)
    return nb_ratio > threshold
