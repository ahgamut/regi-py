"""Self-play / eval plumbing for the AlphaDouZero (ADZ) pipeline.

The candidate-scoring analogue of ``rl.training``: ADZ self-play
(``adz_run_single_game``), the brute late-game path (``adz_run_brute_game``), and
evaluation (``adz_test_model`` / ``adz_improved_gameplay``). It reuses everything
paradigm-agnostic from ``rl.training`` (``run_epoch`` via the shared
``calculate_loss(data, y_hat)`` refactor, ``get_split_optimizer``, ``EndGameLog``,
``drain``, ``total_enemy_hp``, the self-play temperature/discount) and only forks
the pieces that touch the ADZ node/net contract.
"""
import random
import sys
import time

#
import torch
import numpy as np

from regi_py import GameState, DummyLog, seed
from regi_py.strats import RandomStrategy, BruteSamplingStrategy
from regi_py.rl.features import candidate_semantics
from regi_py.rl.adz.explorer import (
    ADZNode,
    ADZNodeInfo,
    ADZDirectStrategy,
    ADZExplorerStrategy,
    adz_simulate_node,
    trimmed_history,
)
from regi_py.rl.training import (
    EndGameLog,
    SELFPLAY_TEMP_MOVES,
    VALUE_DISCOUNT,
    _sample_selfplay_child,
    sample_teammate,
)
from regi_py.rl.utils import enemy_hp_left, hp_loss_penalty


def adz_run_single_game(tid, i, net, num_bots, num_iterations):
    """One ADZ self-play game: candidate-scored MCTS from a fresh ``initialize()``,
    visit-count temperature for the opening then greedy, value target discounted by
    distance to the terminal position. Returns the net's training tensors."""
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
    node = ADZNode(start_phase, net=net, history=[], prior=1.0, trim=False)
    s0 = enemy_hp_left(node.root_phase)
    move_num = 0
    while node.root_phase.game_endvalue == 0:
        # root-only exploration noise (attack AND defense), then search
        node.add_dirichlet_noise()
        adz_simulate_node(node, num_iterations)
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
    # discount the outcome back toward earlier moves (see VALUE_DISCOUNT)
    last = len(history) - 1
    for j, info in enumerate(history):
        info.value = reward * (VALUE_DISCOUNT ** (last - j))
    print(f"{tid},{i},p{len(history)},{s0},{s1},{dt:.4f}s,{win}", file=sys.stderr)
    return type(net).tensorify_training(history)


class RecordingADZBruteStrategy(BruteSamplingStrategy):
    """``BruteSamplingStrategy`` that records, per decision, the plain values an
    ``ADZNodeInfo`` needs -- NEVER a ``PhaseInfo`` reference (the dangling-reference
    rule; see ``rl.training.RecordingBruteStrategy``).

    Unlike the AZ recorder (which stored only the played combo), the ADZ record must
    capture each decision's full offered list + per-candidate semantics, because the
    net scores the ragged offer, not a fixed grid. Both are computed here from the
    by-value ``export_phaseinfo`` copy (a stable snapshot) and kept as plain
    numpy/ints; the history window is rebuilt post-game in ``adz_infos_from_game``.
    """

    def __init__(self, iterations=64):
        super().__init__(iterations=iterations)
        # (history_index, [bitwise...], cand_feats(K,F), played_bitwise, attacking)
        self.moves = []

    def _record_and_pick(self, combos, game):
        root_phase = game.export_phaseinfo()  # by-value copy: safe to featurize now
        try:
            ind = self.get_best_move(root_phase, combos)
        except Exception as e:
            print("failed to process moves", e, file=sys.stderr)
            ind = random.randint(0, len(combos) - 1)
        combo = combos[ind]
        idx = len(game.history) - 1  # this decision phase is history[-1] (plain int)
        bitwises = [c.bitwise for c in combos]
        feats = candidate_semantics(root_phase, combos)  # plain np, safe to hold
        self.moves.append(
            (idx, bitwises, feats, combo.bitwise, float(root_phase.phase_attacking))
        )
        return ind

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        return self._record_and_pick(combos, game)

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        return self._record_and_pick(combos, game)


def adz_infos_from_game(game, moves, value, net_cls):
    """Build the ``ADZNodeInfo`` training list AFTER a brute game finishes, from the
    now-stable ``game.history``. One-shot analogue of ``ADZNode.export()``: the
    policy is a one-hot at the played subset over that decision's offered list. The
    window length is ``net_cls.max_history`` (matches its inference window)."""
    hist = list(game.history)
    maxhist = net_cls.max_history
    infos = []
    for idx, bitwises, feats, played_bw, attacking in moves:
        root_phase = hist[idx]
        window = trimmed_history(hist[:idx], root_phase, maxhist)
        K = len(bitwises)
        policy = np.zeros(K, dtype=np.float32)
        pi = bitwises.index(played_bw) if played_bw in bitwises else 0
        if K:
            policy[pi] = 1.0
        infos.append(
            ADZNodeInfo(
                history=window,
                candidates=bitwises,
                cand_feats=feats,
                policy=policy,
                value=value,
                attacking=attacking,
            )
        )
    return infos


def adz_run_brute_game(tid, i, net_cls, num_bots, iterations):
    """Play one brute-sampling game from a random mid-game state; return training
    tensors only if it made large progress (else ``None``). Mirrors
    ``rl.training.run_brute_game`` but records the ADZ candidate record."""
    a = time.time()
    log = EndGameLog()
    strat = RecordingADZBruteStrategy(iterations=iterations)
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
    if s0 - s1 < 130:  # only games that progress a lot are submitted
        return None
    print(f"{tid},{i},b{len(strat.moves)},{s0},{s1},{dt:.4f}s,{win}", file=sys.stderr)
    # score losses by the hp penalty (matching adz_run_single_game) so the brute and
    # ADZ explorers agree on the value target over the states they both visit
    value = 1.0 if win else hp_loss_penalty(s1)
    infos = adz_infos_from_game(game, strat.moves, value=value, net_cls=net_cls)
    if not infos:
        return None
    return net_cls.tensorify_training(infos)


class RecordingADZTeamStrategy(ADZExplorerStrategy):
    """``ADZExplorerStrategy`` that also records each of ITS OWN decisions the way an
    ``ADZNodeInfo`` needs (``(history_index, [bitwise...], cand_feats, played_bitwise,
    attacking)``, plain values). On a team game's NN seat(s); rebuilt post-game by
    ``adz_infos_from_game``. See the team-games design notes."""

    def __init__(self, net, iterations, moves):
        super().__init__(net, iterations=iterations, trim=False)
        self.moves = moves

    def _record(self, combos, ind, game):
        if ind is None or ind < 0 or not combos:
            return
        root_phase = game.export_phaseinfo()  # by-value copy: safe to featurize now
        idx = len(game.history) - 1  # decision phase is history[-1] (plain int)
        bitwises = [c.bitwise for c in combos]
        feats = candidate_semantics(root_phase, combos)  # plain np, safe to hold
        self.moves.append(
            (idx, bitwises, feats, combos[ind].bitwise, float(root_phase.phase_attacking))
        )

    def getAttackIndex(self, combos, player, yield_allowed, game):
        ind = super().getAttackIndex(combos, player, yield_allowed, game)
        self._record(combos, ind, game)
        return ind

    def getDefenseIndex(self, combos, player, damage, game):
        ind = super().getDefenseIndex(combos, player, damage, game)
        self._record(combos, ind, game)
        return ind


def adz_run_team_game(tid, i, net, num_bots, iterations):
    """ADZ analogue of ``rl.training.run_team_game``: full cooperative game, 1-2
    candidate-scored-MCTS seats + non-NN teammates, training data from the NN's
    decisions only. See the team-games design notes."""
    a = time.time()
    log = EndGameLog()
    moves = []
    nn_strat = RecordingADZTeamStrategy(net, iterations, moves)
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
    value = 1.0 if win else hp_loss_penalty(s1)  # flat outcome value (brute-style)
    infos = adz_infos_from_game(game, moves, value=value, net_cls=type(net))
    if not infos:
        return None
    return type(net).tensorify_training(infos)


def adz_test_model(episode, model, num_simulations):
    model.eval()
    log = EndGameLog()
    diffe = []
    for s in range(10):
        game = GameState(log)
        num_players = random.randint(2, 4)
        for i in range(num_players):
            game.add_player(ADZDirectStrategy(model))
        game.initialize()
        game.start_loop()
        diffe.append(log.diffe())
    print("test games:", diffe, file=sys.stderr)
    torch.save(model.state_dict(), f"./weights/model_{model.__mname__}_{episode}.pt")
    print("episode", episode, "saved model", file=sys.stderr)


def adz_improved_gameplay(episode, new_model, old_model, num_simulations, threshold=0.6):
    new_model.eval()
    old_model.eval()
    log1 = EndGameLog()
    log2 = EndGameLog()

    newer_better = 0
    # evaluate with candidate-scored MCTS (fixed 64 iters/move), not the search-free
    # ADZDirectStrategy, so the comparison reflects competitive play
    old_strat = ADZExplorerStrategy(old_model, iterations=64)
    new_strat = ADZExplorerStrategy(new_model, iterations=64)

    for s in range(num_simulations):
        game1 = GameState(log1)
        game2 = GameState(log2)
        #
        num_players = random.randint(2, 4)
        for i in range(num_players):
            game1.add_player(old_strat)
            game2.add_player(new_strat)
        # both games share the SAME starting RNG seed: identical deal, and each loop
        # replays from the same C++/Python rng stream, so the only difference is
        # which model drives the search
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
