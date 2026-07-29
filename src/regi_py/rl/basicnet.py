"""Back-compat shim: ``BasicNet`` moved to ``regi_py.rl.nets.basicnet`` when the
net became pluggable. Existing ``from regi_py.rl.basicnet import BasicNet`` imports
keep working through this re-export.
"""
from regi_py.rl.nets.basicnet import BasicNet  # noqa: F401

__all__ = ["BasicNet"]
