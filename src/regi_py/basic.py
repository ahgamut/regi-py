from regi_py.core import *
import random

class DummyStrategy(BaseStrategy):
    def setup(self, player, game):
        return 0

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        return random.randint(0, len(combos) - 1)

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        return random.randint(0, len(combos) - 1)

def start_game(n_players=2, log=None) -> GameState:
    assert n_players in [2, 3, 4], "only 2, 3, or 4 players"

    strat = DummyStrategy()
    if log is None:
        log = CXXConsoleLog()
    game = GameState(log)

    for i in range(n_players):
        game.add_player(strat)
    game.initialize()
    return game
