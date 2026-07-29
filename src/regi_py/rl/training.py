"""Generic trainer plumbing for the AlphaZero pipeline.

Lifted out of ``trainers/az_trainer.py`` so that module keeps only the
multiprocessing orchestration (``submain``/``trainer``/``explorer``/``main``).
Everything here is reusable, single-process, and free of any ``mp`` state:
self-play data generation (``run_single_game``), the optimization step
(``run_epoch``), evaluation (``test_model``/``improved_gameplay``), and small
helpers (``EndGameLog``, ``total_enemy_hp``, ``get_split_optimizer``, ``drain``).
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
from regi_py.strats import RandomStrategy, BruteSamplingStrategy
from regi_py.rl.az_explorer import (
    NetDirectStrategy,
    AZExplorerStrategy,
    AlphaZeroNode,
    AZNodeInfo,
    simulate_node,
)
from regi_py.rl.utils import enemy_hp_left, hp_loss_penalty


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
        evicted = buf.add(obj)
        del obj  # drop child's dict ref
        if evicted is not None:
            del evicted  # retire evicted shard


def run_epoch(model, batch, optimizer):
    # a batch is a tuple of tensors in the model's ``TRAIN_FIELDS`` order (that is
    # how ``ShardBuffer`` packs each shard's ``TensorDataset``)
    data = {k: v.to(model.device) for k, v in zip(type(model).TRAIN_FIELDS, batch)}
    v_hat, k_hat, a_hat = model(data)
    v, k, a = data["value"], data["keepyness"], data["atk_probs"]
    loss, (loss1, loss2, loss3) = model.calculate_loss(
        (v, k, a), (v_hat, k_hat, a_hat), data["attacking"]
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # total plus (policy, value, keepy) components for per-head logging
    return loss.item(), (loss1.item(), loss2.item(), loss3.item())


# self-play move selection: for the first SELFPLAY_TEMP_MOVES moves, sample the
# next move in proportion to child visit counts (AlphaZero temperature tau=1) so
# self-play games diversify and the replay buffer covers more distinct states;
# after that play greedily (tau -> 0) toward the strongest line. Greedy-from-move-1
# collapsed every game onto ~one trajectory.
SELFPLAY_TEMP_MOVES = 12

# discount the game outcome back toward earlier moves: the value target for a
# position d moves before the end is reward * VALUE_DISCOUNT**d. Early play has
# little control over the eventual win/loss, so a raw-outcome target broadcast to
# every position is needlessly high-variance and overconfident on distant states.
VALUE_DISCOUNT = 0.98


def _sample_selfplay_child(node, move_num):
    if move_num >= SELFPLAY_TEMP_MOVES:
        return node.best_child_node
    weights = [c.visits for c in node.children]
    if sum(weights) <= 0:
        return node.best_child_node
    return random.choices(node.children, weights=weights, k=1)[0]


def run_single_game(tid, i, net, num_bots, num_iterations):
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
    node = AlphaZeroNode(start_phase, net=net, history=[], prior=1.0, trim=False)
    s0 = enemy_hp_left(node.root_phase)
    move_num = 0
    while node.root_phase.game_endvalue == 0:
        # root-only exploration noise: mix fresh Dirichlet into this move's
        # search root before searching (self-play only, not competitive play)
        node.add_dirichlet_noise()
        simulate_node(node, num_iterations)
        history.append(node.export())
        child = _sample_selfplay_child(node, move_num)
        child.parent = None
        node = child
        move_num += 1
    history.append(node.export())
    win = node.root_phase.game_endvalue == 1
    s1 = enemy_hp_left(node.root_phase)
    #
    dt = time.time() - a
    reward = 1.0 if win else hp_loss_penalty(s1)
    # the last move keeps the full reward; each earlier move is discounted by its
    # distance to the terminal position (see VALUE_DISCOUNT)
    last = len(history) - 1
    for j, info in enumerate(history):
        info.value = reward * (VALUE_DISCOUNT ** (last - j))
    print(f"{tid},{i},p{len(history)},{s0},{s1},{dt:.4f}s,{win}", file=sys.stderr)
    return type(net).tensorify_training(history)


class RecordingBruteStrategy(BruteSamplingStrategy):
    """``BruteSamplingStrategy`` that records, per decision, ONLY the index of the
    decision phase in ``game.history`` plus the played move -- all as plain values,
    never a ``PhaseInfo`` reference.

    ``game.history`` is a pybind view over a live ``std::vector<PhaseInfo>``: its
    elements are references that DANGLE the moment the vector reallocates on the
    next appended phase. The old design held those references (a window per move)
    for the whole game and then segfaulted in ``to_string`` at tensorify time. Here
    nothing but plain ints/bitwise are kept; the windows are rebuilt from the
    stable post-game history in ``infos_from_game``. One instance is shared by all
    players (Regicide is cooperative: every decision in a won game earns the win).
    """

    def __init__(self, iterations=64):
        super().__init__(iterations=iterations)
        self.moves = []  # (history_index, combo_bitwise, [part_location, ...])

    def _record_and_pick(self, combos, game):
        root_phase = game.export_phaseinfo()
        # same fallback contract as the parent's getAttackIndex/getDefenseIndex:
        # a brute failure still plays (and records) a concrete index
        try:
            ind = self.get_best_move(root_phase, combos)
        except Exception as e:
            print("failed to process moves", e, file=sys.stderr)
            ind = random.randint(0, len(combos) - 1)
        combo = combos[ind]
        # start_loop records the current phase at the top of the iteration, before
        # the strategy is consulted, so this decision phase is history[-1]; keep
        # only its index (a plain int, safe to hold across the rest of the game)
        idx = len(game.history) - 1
        self.moves.append((idx, combo.bitwise, [c.location for c in combo.parts]))
        return ind

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        return self._record_and_pick(combos, game)

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        return self._record_and_pick(combos, game)


def infos_from_game(game, moves, value, net_cls):
    """Build the ``AZNodeInfo`` training list AFTER a brute game finishes.

    Rebuilds each decision's history window from the now-stable ``game.history``.
    Safe because the game is over (the underlying vector no longer reallocates) and
    tensorify runs before ``run_brute_game`` returns, while ``game`` -- which keeps
    those references alive -- is still in scope.

    One-shot analogue of ``AlphaZeroNode.export()`` (N0 = 1): ``keepyness`` is 1 for
    kept hand cards / 0 for the cards spent in the played combo; ``atk_probs`` is a
    one-hot at the played combo's ComboTable cell on attack phases (defense phases
    keep it all-zero -- the action loss is masked by ``attacking``). The window
    length is ``net_cls.max_history`` (matches its inference window).
    """
    hist = list(game.history)
    maxhist = net_cls.max_history
    infos = []
    for idx, bitwise, part_locs in moves:
        root_phase = hist[idx]
        # dense window ending at this decision, matching NetDirectStrategy's
        # inference window (the last ``maxhist`` phases of game.history)
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
            AZNodeInfo(
                history=window,
                value=value,
                atk_probs=atk_probs,
                keepyness=keepyness,
            )
        )
    return infos


def run_brute_game(tid, i, net_cls, num_bots, iterations):
    """Play one brute-sampling game from a random mid-game state; return training
    tensors only if it was WON (else ``None``).

    ``init_random`` seeds a random mid-game position (partial deck, fewer
    enemies), so brute wins here yield short, diverse *late-game* trajectories --
    the data AZ self-play (always from a fresh ``initialize()``) never produces.
    """
    a = time.time()
    log = EndGameLog()
    strat = RecordingBruteStrategy(iterations=iterations)
    game = GameState(log)
    for _ in range(num_bots):
        game.add_player(strat)
    game.init_random()
    s0 = enemy_hp_left(game.export_phaseinfo())
    if s0 < 130:
        return None
    game.start_loop()
    end_phase = game.export_phaseinfo()
    s1 = enemy_hp_left(end_phase)
    dt = time.time() - a
    win = end_phase.game_endvalue == 1
    large_progress = False
    if s0 - s1 >= 130:
        large_progress = True
    # only games that progress a lot are submitted as training data
    if not large_progress:
        return None
    print(
        f"{tid},{i},b{len(strat.moves)},{s0},{s1},{dt:.4f}s,{win}", file=sys.stderr
    )
    # build the training records AFTER the game, from the now-stable game.history.
    # score losses by the hp penalty (matching run_single_game) so the brute and
    # AZ explorers agree on the value target over the states they both visit --
    # a high-progress loss is not labelled as good as a win.
    value = 1.0 if win else hp_loss_penalty(s1)
    infos = infos_from_game(game, strat.moves, value=value, net_cls=net_cls)
    if not infos:
        return None
    return net_cls.tensorify_training(infos)


def test_model(episode, model, num_simulations):
    model.eval()
    log = EndGameLog()
    diffe = []
    for s in range(10):
        game = GameState(log)
        num_players = random.randint(2, 4)
        for i in range(num_players):
            game.add_player(NetDirectStrategy(model))
        game.initialize()
        game.start_loop()
        diffe.append(log.diffe())
    print("test games:", diffe, file=sys.stderr)
    torch.save(model.state_dict(), f"./weights/model_{model.__mname__}_{episode}.pt")
    print("episode", episode, "saved model", file=sys.stderr)


def improved_gameplay(episode, new_model, old_model, num_simulations, threshold=0.6):
    new_model.eval()
    old_model.eval()
    log1 = EndGameLog()
    log2 = EndGameLog()

    newer_better = 0
    # evaluate with net-guided MCTS (fixed 64 iters/move), not the search-free
    # NetDirectStrategy, so the comparison reflects competitive play
    old_strat = AZExplorerStrategy(old_model, iterations=64)
    new_strat = AZExplorerStrategy(new_model, iterations=64)

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
