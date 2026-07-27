"""A thin, ergonomic wrapper around :class:`regi_py.core.GameState`.

This is purely additive sugar over the C++ bindings: it offers clean
construction, a one-line initializer, and a Pythonic stepping loop while exposing
the wrapped state as ``.state`` and delegating attribute access to it.  Code that
uses ``GameState`` directly is unaffected.
"""

from .core import GameState
from .logging import DummyLog

__all__ = ["Game"]


class Game:
    """Convenience wrapper: ``Game(log).add_player(...); g.start(); g.run()``.

    The underlying :class:`regi_py.core.GameState` is available as ``.state``;
    unknown attributes (``status``, ``players``, ``phase_count``, ...) are read
    through to it, so a ``Game`` can be used almost anywhere a ``GameState`` is.
    """

    def __init__(self, log=None):
        self.log = log if log is not None else DummyLog()
        self.state = GameState(self.log)

    def add_player(self, strategy):
        """Seat ``strategy`` at the next player id. Returns the player id/code."""
        return self.state.add_player(strategy)

    def start(self, source=None):
        """Initialize the game and return the resulting ``GameStatus``.

        ``source`` selects the initializer: ``None`` -> canonical fresh start;
        a ``str`` -> :meth:`GameState.init_string`; anything else (a
        ``PhaseInfo``) -> :meth:`GameState.init_phaseinfo`.
        """
        if source is None:
            return self.state.initialize()
        if isinstance(source, str):
            return self.state.init_string(source)
        return self.state.init_phaseinfo(source)

    def steps(self):
        """Yield the live ``GameState`` before each phase, until the game ends.

        Usage::

            for st in game.steps():
                ...  # inspect st before the phase is played
        """
        while self.state.is_runnable:
            yield self.state
            self.state.step()

    def run(self):
        """Run the game to completion and return the final ``GameState``."""
        self.state.start_loop()
        return self.state

    def export_string(self):
        return self.state.export_string()

    def export_phaseinfo(self):
        return self.state.export_phaseinfo()

    def __getattr__(self, name):
        # Reached only when normal lookup misses; delegate to the wrapped state.
        # Guard ``state`` itself to avoid infinite recursion before __init__ runs.
        if name == "state":
            raise AttributeError(name)
        return getattr(self.state, name)

    def __repr__(self):
        return f"Game(status={self.state.status!r}, players={self.state.num_players})"
