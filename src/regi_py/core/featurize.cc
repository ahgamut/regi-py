#include <featurize.h>
#include <location.h>
#include <combotable.h>
#include <algorithm>

namespace regi
{
    /* static per-location base strength (mirrors rl/features.py _STRENGTH): a card's
     * own strength, joker locations -> 0. Card::fromLocation succeeds for all 0..55. */
    static const std::vector<float> &strengthTable()
    {
        static const std::vector<float> tbl = []
        {
            std::vector<float> t(MAX_CARDS_IN_GAME, 0.0f);
            for (i32 loc = 0; loc < MAX_CARDS_IN_GAME; ++loc)
            {
                Card c;
                if (c.fromLocation(loc)) { t[loc] = static_cast<float>(c.strength()); }
            }
            return t;
        }();
        return tbl;
    }

    std::vector<float> cardCapabilities(const PhaseInfo &phase)
    {
        const std::vector<float> &S = strengthTable();
        std::vector<float> attack(S), defense(S);  // float32 copies
        for (const Enemy &e : phase.enemyPile)
        {
            i32 loc = e.toLocation();
            attack[loc] = static_cast<float>(-std::max(0, e.hp));
            defense[loc] = static_cast<float>(-e.strength());
        }
        std::vector<float> caps(MAX_CARDS_IN_GAME * CAP_CHANNELS);
        for (i32 i = 0; i < MAX_CARDS_IN_GAME; ++i)
        {
            caps[i * CAP_CHANNELS + 0] = attack[i] / CAP_SCALE;  // float32 divide
            caps[i * CAP_CHANNELS + 1] = defense[i] / CAP_SCALE;
        }
        return caps;
    }

    std::vector<float> locationArray(const PhaseInfo &phase, i32 perspective)
    {
        std::shared_ptr<LocationInfo> info = LocationInfo::fromCurrentPlayer(phase, perspective);
        const i32 R = LocationInfo::rows, C = LocationInfo::cols;  // 56, 9
        std::vector<float> out(R * C);
        for (i32 i = 0; i < R; ++i)
        {
            float rowsum = 0.0f;
            for (i32 j = 0; j < C; ++j)
            {
                float v = static_cast<float>(info->get(i, j));
                out[i * C + j] = v;
                rowsum += v;  // float32 accumulation, matches numpy .sum(axis=1)
            }
            for (i32 j = 0; j < C; ++j) { out[i * C + j] = out[i * C + j] / rowsum; }
        }
        return out;
    }

    std::vector<float> usedPileArray(const PhaseInfo &phase)
    {
        std::shared_ptr<ComboTable> tbl = ComboTable::fromPhaseInfo(phase);
        const i32 R = ComboTable::rows, C = ComboTable::cols;  // 56, 22
        const u32 *data = tbl->getData();
        std::vector<float> out(R * C);
        for (i32 k = 0; k < R * C; ++k) { out[k] = static_cast<float>(data[k]); }
        return out;
    }

    std::vector<float> candidateSemantics(const PhaseInfo &phase,
                                          const std::vector<Combo> &combos)
    {
        /* float64 divides then cast to float32 -- matches python scalar `int / 40.0`
         * assigned into a float32 array (NOT a float32 array op). */
        const i32 hp = phase.enemyPile.empty() ? 0 : phase.enemyPile[0].hp;
        std::vector<float> feats(combos.size() * CAND_FEATURE_DIM, 0.0f);
        for (size_t i = 0; i < combos.size(); ++i)
        {
            const Combo &c = combos[i];
            const i32 dmg = phase.comboDamage(c);
            const i32 blk = phase.comboBlock(c);
            float *row = &feats[i * CAND_FEATURE_DIM];
            row[0] = static_cast<float>(static_cast<double>(dmg) / 40.0);
            row[1] = static_cast<float>(static_cast<double>(blk) / 40.0);
            row[2] = static_cast<float>(static_cast<double>(c.getBaseDamage()) / 40.0);
            row[3] = static_cast<float>(static_cast<double>(c.getBaseDefense()) / 40.0);
            row[4] = c.valid(true) != 0 ? 1.0f : 0.0f;
            row[5] = static_cast<float>(static_cast<double>(c.parts.size()) / 7.0);
            row[6] = c.getBitrep() == 0 ? 1.0f : 0.0f;
            row[7] = (hp > 0 && dmg >= hp) ? 1.0f : 0.0f;
            row[8] = hp > 0
                         ? static_cast<float>(std::min(
                               static_cast<double>(dmg) / static_cast<double>(hp), 1.0))
                         : 0.0f;
        }
        return feats;
    }

    std::vector<float> fuseCardTokens(const std::vector<PhaseInfo> &phases, i32 perspective)
    {
        const i32 W = static_cast<i32>(phases.size());  // window
        const i32 LC = LocationInfo::cols;              // 9
        const i32 UC = ComboTable::cols;                // 22
        const i32 rowlen = W * FEATURE_WIDTH;
        std::vector<float> out(MAX_CARDS_IN_GAME * rowlen, 0.0f);
        for (i32 f = 0; f < W; ++f)
        {
            std::vector<float> loc = locationArray(phases[f], perspective);
            std::vector<float> usp = usedPileArray(phases[f]);
            std::vector<float> cap = cardCapabilities(phases[f]);
            for (i32 card = 0; card < MAX_CARDS_IN_GAME; ++card)
            {
                float *dst = &out[card * rowlen + f * FEATURE_WIDTH];
                for (i32 k = 0; k < LC; ++k) { dst[k] = loc[card * LC + k]; }
                for (i32 k = 0; k < UC; ++k) { dst[LC + k] = usp[card * UC + k]; }
                for (i32 k = 0; k < CAP_CHANNELS; ++k)
                {
                    dst[LC + UC + k] = cap[card * CAP_CHANNELS + k];
                }
            }
        }
        return out;
    }
}
