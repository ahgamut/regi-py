from .core import Suit, Entry, SuitPower, GameStatus, EndGameReason
from .core import Card, Enemy, Combo
from .core import Player
from .core import BaseLog, CXXConsoleLog
from .core import GameState
from .core import PhaseInfo
from .core import LocationInfo, ComboTable
from .core import seed, cards_bitwise
from .game import Game
from .combomap import cell_of_bitwise, bitwise_of_cell, bitwise_to_cell_map
from .logging import JSONBaseLog, JSONLog, RegiEncoder, DummyLog
from .logging import JSONArrayWriter, write_json_array

#
from .strats import STRATEGY_LIST as strat1

__all__ = [
    # enums / constants
    "Suit", "Entry", "SuitPower", "GameStatus", "EndGameReason",
    # value objects
    "Card", "Enemy", "Combo", "Player", "PhaseInfo",
    "LocationInfo", "ComboTable",
    # engine
    "GameState", "Game", "seed",
    # bitwise helpers
    "cards_bitwise", "cell_of_bitwise", "bitwise_of_cell", "bitwise_to_cell_map",
    # logging
    "BaseLog", "CXXConsoleLog", "JSONBaseLog", "JSONLog", "DummyLog",
    "RegiEncoder", "JSONArrayWriter", "write_json_array",
    # helpers
    "get_strategy_map",
]

def get_strategy_map(rl_mods=False):
    all_strats = strat1
    try:
        if rl_mods:
            from .rl import STRATEGY_LIST as strat2

            all_strats = strat1 + strat2
    except ImportError:
        pass

    return {cls.__strat_name__: cls for cls in all_strats}
