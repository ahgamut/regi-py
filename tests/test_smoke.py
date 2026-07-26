"""Smoke tests: the extension imports and a game can be played to completion."""
from regi_py.core import GameState, GameStatus


def test_extension_imports():
    import regi_py.core as core

    assert hasattr(core, "GameState")
    assert hasattr(core, "PhaseInfo")


def test_play_random_game(game):
    assert game.status == GameStatus.RUNNING
    assert game.num_players == 2
    game.start_loop()
    assert game.status == GameStatus.ENDED


def test_export_roundtrip(phase):
    from regi_py.core import PhaseInfo

    s = phase.to_string()
    assert PhaseInfo.from_string(s).to_string() == s
