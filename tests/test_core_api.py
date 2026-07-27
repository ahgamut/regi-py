"""Characterization tests for the core C++/Python value objects and game lifecycle.

These lock the *current* behavior of the pybind11 bindings so later refactors
(serialization unification, RNG, the get_expansion_at rewrite) cannot change it
unnoticed.
"""
import pytest

import regi_py.core as core
from regi_py.core import (
    Card,
    GameState,
    RandomStrategy,
    DamageStrategy,
    BaseStrategy,
    Suit,
    Entry,
    GameStatus,
    EndGameReason,
)
from regi_py.logging import DummyLog

from conftest import make_game


# --------------------------------------------------------------------------- #
# constants and enums
# --------------------------------------------------------------------------- #
def test_max_cards_constant():
    assert core.MAX_CARDS_IN_GAME == 56


def test_suit_values():
    # GLITCH is gone; the four real suits are 0..3
    assert not hasattr(Suit, "GLITCH")
    assert int(Suit.CLUBS) == 0
    assert int(Suit.DIAMONDS) == 1
    assert int(Suit.HEARTS) == 2
    assert int(Suit.SPADES) == 3


def test_entry_values():
    assert int(Entry.JOKER) == 0
    assert int(Entry.ACE) == 1
    assert int(Entry.KING) == 13
    # contiguous 0..13
    assert [int(e) for e in (Entry.TWO, Entry.TEN, Entry.JACK, Entry.QUEEN)] == [2, 10, 11, 12]


def test_gamestatus_values():
    assert (int(GameStatus.LOADING), int(GameStatus.RUNNING), int(GameStatus.ENDED)) == (0, 1, 2)


def test_endgamereason_values():
    assert int(EndGameReason.INVALID_START_PLAYER_COUNT) == 0
    assert int(EndGameReason.INVALID_START_PLAYER_SETUP) == 1
    assert int(EndGameReason.NO_ENEMIES) == 2
    assert int(EndGameReason.PLAYER_DEAD) == 6


# --------------------------------------------------------------------------- #
# Card: location encoding (the single canonical card encoding; no more index)
# --------------------------------------------------------------------------- #
def test_card_location_is_the_only_encoding():
    # the index encoding has been removed entirely
    assert not hasattr(Card.from_location(1), "index")


def test_card_from_location_roundtrip_all():
    # every one of the 56 locations decodes to a distinct card and round-trips
    seen = set()
    for loc in range(core.MAX_CARDS_IN_GAME):
        card = Card.from_location(loc)
        assert card.location == loc
        assert card.location == int(card.entry) + 14 * int(card.suit)
        seen.add((int(card.entry), int(card.suit)))
    assert len(seen) == core.MAX_CARDS_IN_GAME  # bijection


def test_special_joker_slots():
    # the four JOKER-entry cards are the special slots (location % 14 == 0)
    yld = Card.from_location(0)
    resign = Card.from_location(14)
    j1 = Card.from_location(28)
    j2 = Card.from_location(42)
    assert yld.entry == Entry.JOKER and yld.suit == Suit.CLUBS and yld.is_yield
    assert resign.entry == Entry.JOKER and resign.suit == Suit.DIAMONDS and resign.is_resign
    assert j1.entry == Entry.JOKER and j1.suit == Suit.HEARTS  # real joker 1
    assert j2.entry == Entry.JOKER and j2.suit == Suit.SPADES  # real joker 2
    # the two real jokers are distinct (no longer collapsed)
    assert j1 != j2 and j1.location == 28 and j2.location == 42
    # real jokers are neither yield nor resign
    assert not (j1.is_yield or j1.is_resign or j2.is_yield or j2.is_resign)


@pytest.mark.parametrize("loc", [-1, 56, 100])
def test_card_from_location_invalid_raises(loc):
    with pytest.raises(Exception):
        Card.from_location(loc)


def test_card_location_matches_entry_suit():
    # location == entry + 14 * suit for every card dealt into a game
    game = make_game(4)
    cards = [c for p in game.players for c in p.cards]
    cards += list(game.draw_pile) + list(game.discard_pile)
    assert cards, "expected some cards to be dealt"
    for c in cards:
        assert 0 <= c.location < core.MAX_CARDS_IN_GAME
        assert c.location == int(c.entry) + 14 * int(c.suit)


def test_card_strength():
    assert Card.from_location(1).strength == 1  # A of clubs (location 1)
    assert Card.from_location(55).strength == 20  # K of spades (location 55)
    # a jack/queen/king have fixed strengths regardless of numeric entry
    cards = [Card.from_location(loc) for loc in range(core.MAX_CARDS_IN_GAME)]
    for c in cards:
        if c.entry == Entry.KING:
            assert c.strength == 20
        elif c.entry == Entry.QUEEN:
            assert c.strength == 15
        elif c.entry == Entry.JACK:
            assert c.strength == 10


def test_card_ordering_and_hash():
    a_clubs = Card.from_location(1)
    a_clubs2 = Card.from_location(1)
    k_spades = Card.from_location(55)
    assert a_clubs == a_clubs2
    assert a_clubs < k_spades
    assert k_spades > a_clubs
    assert hash(a_clubs) == hash(a_clubs2)
    # usable in a set
    assert len({a_clubs, a_clubs2, k_spades}) == 2


# --------------------------------------------------------------------------- #
# Enemy attributes
# --------------------------------------------------------------------------- #
def test_enemy_attributes():
    game = make_game(2)
    assert len(game.enemy_pile) > 0
    for e in game.enemy_pile:
        assert isinstance(e.hp, int)
        assert e.entry in (Entry.JACK, Entry.QUEEN, Entry.KING)
        assert e.suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
        assert e.strength in (10, 15, 20)


# --------------------------------------------------------------------------- #
# Combo attributes (captured from a running game)
# --------------------------------------------------------------------------- #
class ComboCaptureStrategy(BaseStrategy):
    __strat_name__ = "combo-capture"

    def __init__(self):
        super().__init__()
        self.attack_combos = None

    def setup(self, player, game):
        return 0

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if self.attack_combos is None and len(combos) > 0:
            self.attack_combos = list(combos)
        return 0 if len(combos) > 0 else -1

    def getDefenseIndex(self, combos, player, damage, game):
        return 0 if len(combos) > 0 else -1

    def getRedirectIndex(self, player, game):
        return (game.active_player + 1) % game.num_players


def test_combo_attributes():
    strat = ComboCaptureStrategy()
    game = GameState(DummyLog())
    game.add_player(strat)
    game.add_player(RandomStrategy())
    game.initialize()
    game.start_loop()
    assert strat.attack_combos is not None, "no attack combos were offered"
    # a combo with empty parts represents the "yield" option; real combos have cards.
    for combo in strat.attack_combos:
        assert all(isinstance(c, Card) for c in combo.parts)
        assert isinstance(combo.base_defense, int)
        assert isinstance(combo.bitwise, int)
    assert any(len(combo.parts) >= 1 for combo in strat.attack_combos)


# --------------------------------------------------------------------------- #
# GameState lifecycle and error surfacing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_initialize_is_canonical_start(num_players):
    # Regicide supports 2, 3 or 4 players.
    game = GameState(DummyLog())
    for _ in range(num_players):
        game.add_player(RandomStrategy())
    status = game.initialize()
    assert status == GameStatus.RUNNING
    assert game.num_players == num_players
    # canonical Regicide has 12 royals (4 jacks, 4 queens, 4 kings)
    assert len(game.enemy_pile) == 12


def test_start_loop_reaches_ended():
    game = make_game(2, strategies=[DamageStrategy(), DamageStrategy()])
    assert game.status == GameStatus.RUNNING
    game.start_loop()
    assert game.status == GameStatus.ENDED


class CapturingLog(DummyLog):
    def __init__(self):
        super().__init__()
        self.end_reason = None

    def endgame(self, reason, game):
        self.end_reason = reason


@pytest.mark.parametrize("init_method", ["initialize", "_init_random"])
@pytest.mark.parametrize("num_players", [1, 5, 6])
def test_invalid_player_count_ends_game(num_players, init_method):
    # Only 2-4 players are valid; 1 or 5+ is an invalid setup. Both init paths must
    # end the game cleanly with a reason and must NOT hang (regression: an unset
    # handSize used to loop forever when drawing hands for an invalid count).
    log = CapturingLog()
    game = GameState(log)
    for _ in range(num_players):
        game.add_player(RandomStrategy())
    status = getattr(game, init_method)()
    assert status == GameStatus.ENDED
    assert log.end_reason == EndGameReason.INVALID_START_PLAYER_SETUP


def test_add_player_rejected_after_running():
    game = make_game(2)
    # once RUNNING, adding another player should be refused (returns a negative code)
    ret = game.add_player(RandomStrategy())
    assert ret < 0
    assert game.num_players == 2
