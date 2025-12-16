from .core import Suit, Entry, SuitPower, GameStatus, EndGameReason
from .core import Card, Enemy
from .core import Player
from .core import BaseLog, CXXConsoleLog
from .core import GameState
from .logging import JSONBaseLog, JSONLog, RegiEncoder

#
from .strats import STRATEGY_LIST as strat1

try:
    from .rl import STRATEGY_LIST as strat2

    all_strats = strat1 + strat2
except ImportError:
    all_strats = strat1

STRATEGY_MAP = {cls.__strat_name__: cls for cls in all_strats}
