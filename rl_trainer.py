import argparse

#
from regi_py import GameState
from regi_py import STRATEGY_MAP
from regi_py.rl import MemoryLog, RL1Model


def set_rewards(log, start, end, model):
    last_state = log.memories[end - 1]
    last_state["best_future"] = model.predict(last_state).max()
    if last_state["remaining"] == 0:
        last_state["reward"] = 100
        last_state["best_from_here"] = 100
        last_state["best_future"] = 0
    else:
        last_state["reward"] = -100
        last_state["best_from_here"] = -100

    for i in range((end - 2), (start - 1), -1):
        cur_state = log.memories[i]
        next_state = log.memories[i + 1]
        cur_state["reward"] = cur_state["remaining"] - next_state["remaining"]
        cur_state["best_future"] = next_state["best_from_here"]
        cur_state["best_from_here"] = model.predict(cur_state).max()


def basic_game(strats, log, model):
    n_players = len(strats)
    assert n_players in [2, 3, 4], "only 2, 3, or 4 players"
    print("starting game with bots using", strats)

    start = len(log.memories)
    game = GameState(log)
    for i in range(n_players):
        cls = STRATEGY_MAP[strats[i]]
        cls = log.record(cls)
        obj = cls()
        if strats[i] == "rl1":
            obj.model = model
        game.add_player(obj)
    game.initialize()
    game.start_loop()
    end = len(log.memories)
    set_rewards(log, start, end, model)


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
    d = parser.parse_args()

    log = MemoryLog(N=100)
    model = RL1Model()
    while len(log.memories) <= log.N:
        print(len(log.memories), "memories")
        basic_game(d.bots, log=log, model=model)

if __name__ == "__main__":
    main()
