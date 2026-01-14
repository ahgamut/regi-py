from .core import Suit, Entry, SuitPower, GameStatus, EndGameReason
from .core import Card, Enemy
from .core import Player
from .core import BaseLog, CXXConsoleLog
from .core import GameState
from .logging import JSONBaseLog, JSONLog, RegiEncoder, DummyLog

#
from .strats import STRATEGY_LIST as strat1

def get_strategy_map(rl_mods=False):
    all_strats = strat1
    try:
        if rl_mods:
            from .rl import STRATEGY_LIST as strat2

            all_strats = strat1 + strat2
    except ImportError:
        pass

    return {cls.__strat_name__: cls for cls in all_strats}
