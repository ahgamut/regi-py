from regi_py.core import BaseStrategy
from regi_py.core import RandomStrategy
from regi_py.core import DamageStrategy
from .basic import DummyStrategy
from .preserve import PreserveStrategy

STRATEGY_MAP = {
    cls.__strat_name__: cls
    for cls in [RandomStrategy, DamageStrategy, DummyStrategy, PreserveStrategy]
}
