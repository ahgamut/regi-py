import argparse

#
from regi_py import JSONLog, CXXConsoleLog, GameState
from regi_py.strats import DummyStrategy


def basic_game(n_players=2, log=None) -> GameState:
    assert n_players in [2, 3, 4], "only 2, 3, or 4 players"

    strat = DummyStrategy()
    if log is None:
        log = CXXConsoleLog()
    game = GameState(log)

    for i in range(n_players):
        game.add_player(strat)
    game.initialize()
    return game


def main():
    parser = argparse.ArgumentParser("basic-regi")
    parser.add_argument(
        "-n", "--num-players", default=2, type=int, help="number of players"
    )
    parser.add_argument("-o", "--output-json", default=None, help="Log Output to JSON")
    d = parser.parse_args()

    if d.output_json is not None:
        log = JSONLog(d.output_json)
    else:
        log = CXXConsoleLog()

    game = basic_game(d.num_players, log=log)
    game.start_loop()


if __name__ == "__main__":
    main()
