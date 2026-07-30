"""Registry of AlphaZero net architectures, keyed by ``__mname__``.

``get_net(name)`` returns the class for ``--net <name>``; ``net_names()`` lists the
registered names. Each concrete net module is imported here so it is registered on
``import regi_py.rl.az.nets`` (import order is controlled centrally to avoid cycles).
"""
from regi_py.rl.az.nets.base import BaseNet

_REGISTRY = {}


def register(cls):
    """Register a ``BaseNet`` subclass under its ``__mname__`` (usable as a
    decorator; also called explicitly below)."""
    _REGISTRY[cls.__mname__] = cls
    return cls


def get_net(name):
    try:
        return _REGISTRY[name]
    except KeyError:
        raise SystemExit(f"unknown --net {name!r}; choices: {net_names()}")


def net_names():
    return sorted(_REGISTRY)


# concrete architectures (import + register here; add future nets alongside)
from regi_py.rl.az.nets.basicnet import BasicNet  # noqa: E402
from regi_py.rl.az.nets.attntrunk import AttnTrunkNet  # noqa: E402
from regi_py.rl.az.nets.cardtx import CardTransformerNet  # noqa: E402
from regi_py.rl.az.nets.percardmlp import PerCardMLPNet  # noqa: E402
from regi_py.rl.az.nets.mixer import MixerNet  # noqa: E402
from regi_py.rl.az.nets.movetoken import MoveTokenNet  # noqa: E402

register(BasicNet)
register(AttnTrunkNet)
register(CardTransformerNet)
register(PerCardMLPNet)
register(MixerNet)
register(MoveTokenNet)

__all__ = [
    "BaseNet",
    "BasicNet",
    "AttnTrunkNet",
    "CardTransformerNet",
    "PerCardMLPNet",
    "MixerNet",
    "MoveTokenNet",
    "register",
    "get_net",
    "net_names",
]
