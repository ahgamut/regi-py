from regi_py import GameState, Player, RandomStrategy, CXXConsoleLog

strat = RandomStrategy()
log = CXXConsoleLog()
players = [Player(strat), Player(strat)]

print(players[0])

game = GameState(log, players)
game.initialize()
# game.start_loop()
