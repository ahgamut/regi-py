"""Registry of AlphaDouZero (ADZ) candidate-scoring net architectures, keyed by
``__mname__``.

Deliberately SEPARATE from ``rl.az.nets`` (the card-space AZ registry): an ADZ net
and an AZ net have incompatible predict/search/tensorify contracts, so
``adz_trainer --net`` must never be able to instantiate a card-space ``BaseNet``
and vice versa. ``get_adz_net(name)`` returns the class for ``--net <name>``;
``adz_net_names()`` lists the registered names.
"""
from regi_py.rl.adz.nets.base import CandidateBaseNet

_ADZ_REGISTRY = {}


def register_adz(cls):
    """Register a ``CandidateBaseNet`` subclass under its ``__mname__``."""
    _ADZ_REGISTRY[cls.__mname__] = cls
    return cls


def get_adz_net(name):
    try:
        return _ADZ_REGISTRY[name]
    except KeyError:
        raise SystemExit(f"unknown --net {name!r}; choices: {adz_net_names()}")


def adz_net_names():
    return sorted(_ADZ_REGISTRY)


# concrete architectures (imported here so their @register_adz runs on
# ``import regi_py.rl.adz.nets``; add future ADZ nets alongside)
from regi_py.rl.adz.nets.douzero import MultiHotActionNet  # noqa: E402,F401

__all__ = [
    "CandidateBaseNet",
    "MultiHotActionNet",
    "register_adz",
    "get_adz_net",
    "adz_net_names",
]
