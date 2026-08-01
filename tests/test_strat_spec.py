"""Torch-free coverage of the unified strategy-spec builder (``regi_py.strat_spec``).

The NN-net branch needs torch and is exercised in the user's torch env; here we cover
the parser grammar, ``spec_uses_nn`` (which must stay torch-free), and every torch-free
build branch (zero-arg names, dashed zero-arg names, and ``brute-N`` / ``mcts-N``).
"""

import pytest

from regi_py import build_strategy, spec_uses_nn, get_strategy_map
from regi_py.strat_spec import parse_spec
from regi_py.core import BaseStrategy


def test_parse_spec_string_forms():
    assert parse_spec("random") == ("random", None, None)
    assert parse_spec("brute-128") == ("brute", 128, None)
    assert parse_spec("mcts-64") == ("mcts", 64, None)
    # a dashed zero-arg name: the tail is not an int, so it is NOT split off as iters.
    assert parse_spec("sub-random") == ("sub-random", None, None)
    assert parse_spec("trim-random") == ("trim-random", None, None)


def test_parse_spec_dict_form():
    assert parse_spec({"name": "adzmulti", "iters": 64, "weights": "w.pt"}) == (
        "adzmulti",
        64,
        "w.pt",
    )
    assert parse_spec({"name": "basic"}) == ("basic", None, None)


@pytest.mark.parametrize("bad", ["", 5, None, {"iters": 3}])
def test_parse_spec_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_spec(bad)


def test_spec_uses_nn_is_torch_free_and_correct():
    # torch-free builtins / zero-arg strategies are NOT NN
    for spec in ["random", "damage", "preserve", "sub-random", "brute-128", "mcts-64", "CDHS"]:
        assert spec_uses_nn(spec) is False
    # a net architecture name (dict form) IS NN
    assert spec_uses_nn({"name": "adzmulti", "iters": 0, "weights": "w.pt"}) is True
    # a bare NN net name string is also NN
    assert spec_uses_nn("adzmulti-64") is True


@pytest.mark.parametrize("name", ["random", "damage", "preserve", "dummy", "sub-random", "CDHS"])
def test_build_zero_arg_strategies(name):
    strat = build_strategy(name)
    assert isinstance(strat, BaseStrategy)


def test_build_search_strategies_carry_iters():
    brute = build_strategy("brute-256")
    assert isinstance(brute, BaseStrategy)
    assert brute.iterations == 256

    mcts = build_strategy("mcts-32")
    assert isinstance(mcts, BaseStrategy)
    assert mcts.iterations == 32


def test_build_search_defaults_when_no_iters():
    # bare "brute" (no count) -> default iterations, still a real strategy
    assert isinstance(build_strategy("brute"), BaseStrategy)


def test_every_zero_arg_name_builds():
    # the whole torch-free map is buildable by name
    for name in get_strategy_map(rl_mods=False):
        assert isinstance(build_strategy(name), BaseStrategy)
