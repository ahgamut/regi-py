from regi_py.core import BaseStrategy, Suit
from dataclasses import dataclass
import random

SUITPREF_MAP = {
    Suit.GLITCH: "X",
    Suit.CLUBS: "C",
    Suit.DIAMONDS: "D",
    Suit.HEARTS: "H",
    Suit.SPADES: "S",
}


@dataclass(slots=True, frozen=True)
class SuitValuation:
    score: float
    pref: float
    attacking: bool

    def __eq__(self, other):
        return (self.score == other.score) and (self.pref == other.pref)

    def __ne__(self, other):
        return (self.score != other.score) or (self.pref != other.pref)

    def __lt__(self, other):
        raw = None
        if abs(self.score - other.score) <= 2:
            return self.pref < other.pref
        return self.score < other.score

    def __le__(self, other):
        return (self < other) or (self == other)

    def __gt__(self, other):
        if abs(self.score - other.score) <= 2:
            return self.pref > other.pref
        return self.score > other.score

    def __ge__(self, other):
        return (self > other) or (self == other)


class SuitPrefStrategy(BaseStrategy):
    __strat_name__ = "suitpref"
    __suit_order__ = None

    def setup(self, player, game):
        return 0

    def get_preference(self, combo, is_attacking):
        sval = list(self.__suit_order__)
        pref = 0
        for c in combo.parts:
            if c.suit == Suit.GLITCH:
                pref = random.randint(0, 16)
                break
            ind = sval.index(SUITPREF_MAP[c.suit])
            if is_attacking:
                low = 2 ** (3 - ind)
                hi = 2**(4 - ind)
            else:
                low = 2 ** (ind)
                hi = 2**(ind + 1)
            pref = max(pref, random.randint(low, hi))

        return pref

    def get_best_combo(self, player, combos, game, is_attacking):
        # hello = "attack" if is_attacking else "defend"
        # print("sorting combos for", hello, "via", self.__suit_order__)
        e = game.enemy_pile[0]
        scores = dict()

        for ind, x in enumerate(combos):
            # as close to enemy hp as possible
            if is_attacking:
                d0 = game.get_combo_damage(e, x)
            else:
                d0 = game.get_combo_block(e, x) + sum(c.entry.value for c in x.parts)
            pref = self.get_preference(x, is_attacking)
            scores[ind] = SuitValuation(d0, pref, is_attacking)

        faraways = []
        for k, v in scores.items():
            if len(scores) - len(faraways) < 2:
                break
            if is_attacking:
                if abs(e.hp - v.score) > 5:
                    faraways.append(k)
            else:
                if abs(e.strength - v.score) > 5:
                    faraways.append(k)

        for k in faraways:
            scores.pop(k)

        # for k,v in scores.items():
        #    print(combos[k], f"{hello}: {v.score}", f"preference: {v.pref}")

        scores2 = list(scores.keys())
        best_ind = 0
        for i in range(1, len(scores2)):
            b2i = scores2[i]
            b2b = scores2[best_ind]
            if scores[b2i] > scores[b2b]:
                best_ind = i
            elif scores[b2i] == scores[b2b]:
                if random.random() > 0.5:
                    best_ind = i
        # print("best combo for", hello, "is", combos[scores2[best_ind]])
        return scores2[best_ind]

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        return self.get_best_combo(player, combos, game, True)

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        return self.get_best_combo(player, combos, game, False)


class CDHSPref(SuitPrefStrategy):
    __suit_order__ = "CDHS"
    __strat_name__ = "CDHS"


class CDSHPref(SuitPrefStrategy):
    __suit_order__ = "CDSH"
    __strat_name__ = "CDSH"


class CHDSPref(SuitPrefStrategy):
    __suit_order__ = "CHDS"
    __strat_name__ = "CHDS"


class CHSDPref(SuitPrefStrategy):
    __suit_order__ = "CHSD"
    __strat_name__ = "CHSD"


class CSDHPref(SuitPrefStrategy):
    __suit_order__ = "CSDH"
    __strat_name__ = "CSDH"


class CSHDPref(SuitPrefStrategy):
    __suit_order__ = "CSHD"
    __strat_name__ = "CSHD"


#


class DCHSPref(SuitPrefStrategy):
    __suit_order__ = "DCHS"
    __strat_name__ = "DCHS"


class DCSHPref(SuitPrefStrategy):
    __suit_order__ = "DCSH"
    __strat_name__ = "DCSH"


class DHCSPref(SuitPrefStrategy):
    __suit_order__ = "DHCS"
    __strat_name__ = "DHCS"


class DHSCPref(SuitPrefStrategy):
    __suit_order__ = "DHSC"
    __strat_name__ = "DHSC"


class DSCHPref(SuitPrefStrategy):
    __suit_order__ = "DSCH"
    __strat_name__ = "DSCH"


class DSHCPref(SuitPrefStrategy):
    __suit_order__ = "DSHC"
    __strat_name__ = "DSHC"


#


class HCDSPref(SuitPrefStrategy):
    __suit_order__ = "HCDS"
    __strat_name__ = "HCDS"


class HCSDPref(SuitPrefStrategy):
    __suit_order__ = "HCSD"
    __strat_name__ = "HCSD"


class HDCSPref(SuitPrefStrategy):
    __suit_order__ = "HDCS"
    __strat_name__ = "HDCS"


class HDSCPref(SuitPrefStrategy):
    __suit_order__ = "HDSC"
    __strat_name__ = "HDSC"


class HSCDPref(SuitPrefStrategy):
    __suit_order__ = "HSCD"
    __strat_name__ = "HSCD"


class HSDCPref(SuitPrefStrategy):
    __suit_order__ = "HSDC"
    __strat_name__ = "HSDC"


#


class SCDHPref(SuitPrefStrategy):
    __suit_order__ = "SCDH"
    __strat_name__ = "SCDH"


class SCHDPref(SuitPrefStrategy):
    __suit_order__ = "SCHD"
    __strat_name__ = "SCHD"


class SDCHPref(SuitPrefStrategy):
    __suit_order__ = "SDCH"
    __strat_name__ = "SDCH"


class SDHCPref(SuitPrefStrategy):
    __suit_order__ = "SDHC"
    __strat_name__ = "SDHC"


class SHCDPref(SuitPrefStrategy):
    __suit_order__ = "SHCD"
    __strat_name__ = "SHCD"


class SHDCPref(SuitPrefStrategy):
    __suit_order__ = "SHDC"
    __strat_name__ = "SHDC"


AllPrefs = [
    CDHSPref,
    CDSHPref,
    CHDSPref,
    CHSDPref,
    CSDHPref,
    CSHDPref,
    DCHSPref,
    DCSHPref,
    DHCSPref,
    DHSCPref,
    DSCHPref,
    DSHCPref,
    HCDSPref,
    HCSDPref,
    HDCSPref,
    HDSCPref,
    HSCDPref,
    HSDCPref,
    SCDHPref,
    SCHDPref,
    SDCHPref,
    SDHCPref,
    SHCDPref,
    SHDCPref,
]
