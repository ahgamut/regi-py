import argparse

#
from regi_py import basic_game


def main():
    parser = argparse.ArgumentParser("basic-regi")
    parser.add_argument(
        "-n", "--num-players", default=2, type=int, help="number of players"
    )
    d = parser.parse_args()

    game = basic_game(d.num_players)
    game.start_loop()


if __name__ == "__main__":
    main()
