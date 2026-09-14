"""Bit-for-bit parity between the C++ featurizer kernels (``core.features_*``) and
the original numpy featurization glue.

The nets were trained on the numpy features, so the C++ kernels that now back
``rl/features.py`` (and the browser Embind build) must reproduce the numpy output
*exactly* in float32 -- not merely within a tolerance. This test carries its own
snapshot of the numpy reference (``_golden_*``) so it stays a valid oracle even as
``rl/features.py`` becomes a thin delegator to the C++ kernels.

Torch-free: needs only the built C++ extension + numpy.
"""
import numpy as np
import pytest

core = pytest.importorskip("regi_py.core", reason="regi_py.core extension not built")

from regi_py.core import GameState, RandomStrategy, PhaseInfo, Card  # noqa: E402
from regi_py.core import MAX_CARDS_IN_GAME  # noqa: E402
from regi_py.logging import DummyLog  # noqa: E402
from regi_py.strats.phase_utils import PhaseExpander  # noqa: E402

# ---- golden numpy reference (snapshot of the original rl/features.py glue) ----

_CAP_SCALE = 40.0
_CAP_CHANNELS = 2
_CAND_FEATURE_DIM = 9
_PARTS_SCALE = 7.0

_STRENGTH = np.zeros(MAX_CARDS_IN_GAME, dtype=np.float32)
for _loc in range(MAX_CARDS_IN_GAME):
    try:
        _STRENGTH[_loc] = Card.from_location(_loc).strength
    except Exception:
        pass


def _golden_card_capabilities(phase):
    attack = _STRENGTH.copy()
    defense = _STRENGTH.copy()
    for enemy in phase.enemy_pile:
        loc = enemy.location
        attack[loc] = -max(0, enemy.hp)
        defense[loc] = -enemy.strength
    caps = np.empty((MAX_CARDS_IN_GAME, _CAP_CHANNELS), dtype=np.float32)
    caps[:, 0] = attack
    caps[:, 1] = defense
    caps /= _CAP_SCALE
    return caps


def _golden_location_array(phase, perspective):
    from regi_py.core import LocationInfo

    loca0 = np.array(LocationInfo.from_current(phase, perspective), dtype=np.float32)
    return loca0 / loca0.sum(axis=1, keepdims=True)


def _golden_used_pile_array(phase):
    from regi_py.core import ComboTable

    return np.array(ComboTable.from_phase(phase), dtype=np.float32)


def _golden_candidate_semantics(phase, combos):
    enemy_hp = phase.enemy_pile[0].hp if len(phase.enemy_pile) else 0
    feats = np.zeros((len(combos), _CAND_FEATURE_DIM), dtype=np.float32)
    for i, c in enumerate(combos):
        dmg = phase.combo_damage(c)
        blk = phase.combo_block(c)
        feats[i, 0] = dmg / _CAP_SCALE
        feats[i, 1] = blk / _CAP_SCALE
        feats[i, 2] = c.base_damage / _CAP_SCALE
        feats[i, 3] = c.base_defense / _CAP_SCALE
        feats[i, 4] = 1.0 if c.can_attack else 0.0
        feats[i, 5] = len(c.parts) / _PARTS_SCALE
        feats[i, 6] = 1.0 if c.bitwise == 0 else 0.0
        feats[i, 7] = 1.0 if (enemy_hp > 0 and dmg >= enemy_hp) else 0.0
        feats[i, 8] = min(dmg / enemy_hp, 1.0) if enemy_hp > 0 else 0.0
    return feats


# ---- phase corpus: attack + defense decisions from real games ----


def _collect_phases(seeded, n_games=30, num_players=2):
    core.seed(seeded)
    attack, defense = [], []
    for _ in range(n_games):
        game = GameState(DummyLog())
        for _ in range(num_players):
            game.add_player(RandomStrategy())
        game.initialize()
        game.start_loop()
        for ph in game.history:
            (attack if ph.phase_attacking else defense).append(ph.to_string())
    return attack, defense


def _assert_exact(got, ref):
    assert got.dtype == np.float32 == ref.dtype
    assert got.shape == ref.shape
    # bit-for-bit: identical float32 bit patterns (NaN-safe identity)
    assert got.tobytes() == ref.tobytes(), (
        "float32 mismatch; max|diff|=%r" % (np.abs(got.astype(np.float64) - ref.astype(np.float64)).max(),)
    )


def test_card_capabilities_parity(seeded):
    attack, defense = _collect_phases(seeded)
    seen = 0
    for s in attack[:200] + defense[:200]:
        phase = PhaseInfo.from_string(s)
        _assert_exact(core.features_card_capabilities(phase), _golden_card_capabilities(phase))
        seen += 1
    assert seen > 100


def test_used_pile_array_parity(seeded):
    attack, defense = _collect_phases(seeded)
    seen = 0
    for s in attack[:200] + defense[:200]:
        phase = PhaseInfo.from_string(s)
        _assert_exact(core.features_used_pile_array(phase), _golden_used_pile_array(phase))
        seen += 1
    assert seen > 100


def test_location_array_parity_all_perspectives(seeded):
    # cover every valid seat index across 2/3/4-player games (perspective must be a
    # real player id: LocationInfo.from_current is UB for out-of-range seats).
    seen = 0
    for num_players in (2, 3, 4):
        attack, defense = _collect_phases(seeded, n_games=12, num_players=num_players)
        for s in attack[:120] + defense[:120]:
            phase = PhaseInfo.from_string(s)
            for persp in range(len(phase.player_cards)):
                _assert_exact(
                    core.features_location_array(phase, persp),
                    _golden_location_array(phase, persp),
                )
            seen += 1
    assert seen > 150


def test_candidate_semantics_parity(seeded):
    attack, defense = _collect_phases(seeded)
    checked = dict(rows=0, lethal=0, block=0, damage=0, empty=0)
    for s in attack[:250] + defense[:250]:
        phase = PhaseInfo.from_string(s)
        combos = PhaseExpander(phase).offered()
        got = core.features_candidate_semantics(phase, combos)
        ref = _golden_candidate_semantics(phase, combos)
        _assert_exact(got, ref)
        if not combos:
            checked["empty"] += 1
            continue
        checked["rows"] += len(combos)
        checked["lethal"] += int((ref[:, 7] == 1.0).any())
        checked["block"] += int((ref[:, 1] != 0).any())
        checked["damage"] += int((ref[:, 0] != 0).any())
    # not vacuous: real damage, blocks, and lethal moves were exercised
    assert checked["rows"] > 500, checked
    assert checked["lethal"] > 0 and checked["block"] > 0 and checked["damage"] > 0, checked


def test_candidate_semantics_empty():
    game = GameState(DummyLog())
    for _ in range(2):
        game.add_player(RandomStrategy())
    game.initialize()
    phase = game.export_phaseinfo()
    got = core.features_candidate_semantics(phase, [])
    assert got.shape == (0, _CAND_FEATURE_DIM) and got.dtype == np.float32
