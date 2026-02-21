from regi_py.core import BaseStrategy
from regi_py.core import RandomStrategy
from regi_py.core import DamageStrategy
from .basic import DummyStrategy
from .preserve import PreserveStrategy
from .suitpref import AllPrefs as AllSuits
from .trim_random import TrimmedRandomStrategy
from .sub_random import SubsetRandomStrategy
from .brute_sampling import BruteSamplingStrategy
from .mcts_explorer import MCTSExplorerStrategy

STRATEGY_LIST = [
    RandomStrategy,
    SubsetRandomStrategy,
    DamageStrategy,
    DummyStrategy,
    PreserveStrategy,
    BruteSamplingStrategy,
] + AllSuits
