from regi_py.core import BaseStrategy
from regi_py.core import RandomStrategy
from regi_py.core import DamageStrategy
from .basic import DummyStrategy
from .preserve import PreserveStrategy
from .suitpref import AllPrefs as AllSuits
from .trim_random import TrimmedRandomStrategy
from .brute_sampling import BruteSamplingStrategy
from .mcts_explorer import MCTSExplorerStrategy

STRATEGY_LIST = [
    RandomStrategy,
    DamageStrategy,
    DummyStrategy,
    PreserveStrategy,
    TrimmedRandomStrategy,
    BruteSamplingStrategy,
    MCTSExplorerStrategy,
] + AllSuits
