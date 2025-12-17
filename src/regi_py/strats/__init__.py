from regi_py.core import BaseStrategy
from regi_py.core import RandomStrategy
from regi_py.core import DamageStrategy
from .basic import DummyStrategy
from .preserve import PreserveStrategy
from .suitpref import AllPrefs as AllSuits

STRATEGY_LIST = [
    RandomStrategy,
    DamageStrategy,
    DummyStrategy,
    PreserveStrategy,
] + AllSuits
