from regi_py import GameState, Player, RandomStrategy, CXXConsoleLog

strat = RandomStrategy()
log = CXXConsoleLog()
game = GameState(log)

game.add_player(strat)
game.add_player(strat)
game.initialize()
game.start_loop()
