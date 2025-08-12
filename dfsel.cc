#include <dfsel.h>

namespace regi
{
    namespace dfsel
    {
        void collectAttack(std::vector<Card> &cards, std::vector<Combo> &combos,
                           bool yieldAllowed, Combo &cur, std::int32_t i)
        {
            if (i >= cards.size()) { return; }
            // try adding cards[i] to the combo
            cur.parts.push_back(Card(cards[i].entry(), cards[i].suit()));
            if (cur.valid(yieldAllowed))
            {
                // if combo is valid, accumulate,
                // and try extending the combo
                combos.push_back(cur);
                for (std::int32_t j = i + 1; j < cards.size(); ++j)
                {
                    collectAttack(cards, combos, yieldAllowed, cur, j);
                }
            }
            cur.parts.pop_back();
        }

        Combo &selectAttack(std::vector<Combo> &combos)
        {
            std::int32_t len = combos.size();
            std::random_device dev;
            std::default_random_engine engine(dev());
            std::int32_t i = engine() % len;
            return combos[i];
        }

        void collectDefense(std::vector<Card> &cards, std::vector<Combo> &combos,
                            std::int32_t damage, Combo &cur, std::int32_t i)
        {
            if (i >= cards.size()) { return; }
            // try adding cards[i] to the combo
            cur.parts.push_back(Card(cards[i].entry(), cards[i].suit()));
            if (cur.getBaseDefense() >= damage)
            {
                // if combo can defend, accumulate
                combos.push_back(cur);
            }
            // try extending the combo
            for (std::int32_t j = i + 1; j < cards.size(); ++j)
            {
                collectAttack(cards, combos, false, cur, j);
            }
            cur.parts.pop_back();
        }

        Combo &selectDefense(std::vector<Combo> &combos)
        {
            std::int32_t len = combos.size();
            std::random_device dev;
            std::default_random_engine engine(dev());
            std::int32_t i = engine() % len;
            return combos[i];
        }

    } /* namespace dfsel */
} /* namespace regi */
