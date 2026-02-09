from regi_py.core import BaseStrategy
from regi_py.core import RandomStrategy
from regi_py.core import DamageStrategy
from .basic import DummyStrategy
from .preserve import PreserveStrategy
from .suitpref import AllPrefs as AllSuits
from .trim_random import TrimmedRandomStrategy

STRATEGY_LIST = [
    RandomStrategy,
    DamageStrategy,
    DummyStrategy,
    PreserveStrategy,
    TrimmedRandomStrategy,
] + AllSuits
