import argparse

#
from regi_py import JSONLog, CXXConsoleLog, GameState
from regi_py import get_strategy_map

STRATEGY_MAP = get_strategy_map()


def basic_game(strats, log=None) -> GameState:
    n_players = len(strats)
    assert n_players in [2, 3, 4], "only 2, 3, or 4 players"
    print("starting game with bots using", strats)

    if log is None:
        log = CXXConsoleLog()
    game = GameState(log)

    for i in range(n_players):
        game.add_player(STRATEGY_MAP[strats[i]]())
    game.initialize()
    return game


def main():
    parser = argparse.ArgumentParser("basic-regi")
    parser.add_argument(
        "-b",
        "--add-bot",
        dest="bots",
        action="append",
        default=[],
        help="bot options: " + ",".join(STRATEGY_MAP),
    )
    parser.add_argument("-o", "--output-json", default=None, help="Log Output to JSON")
    d = parser.parse_args()

    if d.output_json is not None:
        log = JSONLog(d.output_json)
    else:
        log = CXXConsoleLog()

    while len(d.bots) < 2:
        d.bots.append("dummy")

    game = basic_game(d.bots, log=log)
    game.start_loop()


if __name__ == "__main__":
    main()
