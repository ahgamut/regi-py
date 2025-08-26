import argparse

#
from regi_py import basic_game, JSONLog, CXXConsoleLog


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
