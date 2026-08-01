"""Unified strategy-spec builder.

Turns a benchmark *spec* into a live :class:`Strategy` instance.  A spec is one of:

* a bare zero-arg strategy name  -- ``"random"``, ``"damage"``, ``"preserve"``,
  ``"dummy"``, ``"sub-random"``, ``"trim-random"``, a suit-pref like ``"CDHS"`` ...
  (any ``__strat_name__`` in :func:`regi_py.get_strategy_map`).
* a search strategy ``"NAME-ITERS"``  -- ``"brute-128"`` / ``"mcts-64"``.
* an NN net, as a bare ``"NAME-ITERS"`` string **with a weights path supplied out of
  band**, or (the usual benchmark form) a dict ``{"name", "iters", "weights"}``.
  ``iters == 0`` => search-free Direct-net, ``iters > 0`` => net-guided Explorer.

This is the superset of :func:`regi_py.get_strategy_map` (zero-arg torch-free classes)
and :func:`regi_py.rl.make_net_strategy` (NN nets), plus the ``NAME-ITERS`` grammar the
webapp's ``webdriver/common.py:parse_reco_spec`` uses.  It stays **torch-free at
import** -- torch loads lazily only when an NN net is actually built -- so a
brute/mcts/zero-arg run never pulls torch.
"""

# NOTE: ``regi_py.get_strategy_map`` / ``regi_py.rl.make_net_strategy`` are imported
# INSIDE the functions below, not at module top level: this module is re-exported from
# ``regi_py/__init__.py``, so a top-level import would be circular, and deferring the
# ``rl`` import is also what keeps the torch dependency lazy.

# Torch-free builtins that take an iteration count (not in ``get_strategy_map`` as an
# iters-taking entry: ``brute`` is there only zero-arg, ``mcts`` is not there at all).
_SEARCH_BUILTINS = ("brute", "mcts")


def parse_spec(spec):
    """Normalize a spec into ``(name, iters, weights)``.

    ``iters`` is ``None`` when the spec carries no explicit count.  For a string the
    trailing ``-<int>`` is split off as ``iters`` **only when it parses as an int**, so
    dashed zero-arg names (``sub-random``, ``trim-random``) are left intact.
    """
    if isinstance(spec, dict):
        if "name" not in spec:
            raise ValueError(f"strategy spec dict needs a 'name': {spec!r}")
        return spec["name"], spec.get("iters"), spec.get("weights")

    if not isinstance(spec, str) or not spec:
        raise ValueError(f"strategy spec must be a non-empty str or dict; got {spec!r}")

    head, sep, tail = spec.rpartition("-")
    if sep and head:
        try:
            return head, int(tail), None
        except ValueError:
            pass  # e.g. "sub-random" -> tail "random" is not an int; a bare name
    return spec, None, None


def _is_nn(name, iters):
    """True when ``(name, iters)`` names an NN net (not a torch-free builtin)."""
    from regi_py import get_strategy_map

    if name in _SEARCH_BUILTINS:
        return False
    if iters is None and name in get_strategy_map(rl_mods=False):
        return False
    return True


def spec_uses_nn(spec):
    """True when ``spec`` resolves to an NN-net strategy (torch needed to build it).

    Torch-free: it only inspects names, it never imports torch or builds anything.
    Lets a caller decide up front whether the run needs torch configured at all.
    """
    name, iters, _ = parse_spec(spec)
    return _is_nn(name, iters)


def build_strategy(spec):
    """Build a :class:`Strategy` instance from ``spec`` (see the module docstring).

    torch is imported lazily and ONLY when ``spec`` is an NN net (branch 4 below).
    """
    name, iters, weights = parse_spec(spec)

    if name == "brute":
        from regi_py.strats import BruteSamplingStrategy

        return BruteSamplingStrategy() if iters is None else BruteSamplingStrategy(iters)
    if name == "mcts":
        from regi_py.strats.mcts_explorer import MCTSExplorerStrategy

        return MCTSExplorerStrategy() if iters is None else MCTSExplorerStrategy(iters)

    if iters is None:
        from regi_py import get_strategy_map

        strategy_map = get_strategy_map(rl_mods=False)
        if name in strategy_map:
            return strategy_map[name]()  # zero-arg torch-free strategy

    # Anything else is an NN net architecture -> lazy torch build (needs weights).
    from regi_py.rl import make_net_strategy

    return make_net_strategy(name, iters or 0, weights)
