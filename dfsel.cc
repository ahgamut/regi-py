#include <dfsel.h>

namespace regi
{
    namespace dfsel
    {
        void collectAttack(std::vector<Card> &cards, std::vector<Combo> &combos,
                           bool yieldAllowed, Combo &cur, int i)
        {
            if (i >= cards.size()) { return; }
            // try adding cards[i] to the combo
            cur.parts.push_back(Card(cards[i].entry(), cards[i].suit()));
            if (cur.valid(yieldAllowed))
            {
                // if combo is valid, accumulate,
                // and try extending the combo
                combos.push_back(cur);
                for (int j = i + 1; j < cards.size(); ++j)
                {
                    collectAttack(cards, combos, yieldAllowed, cur, j);
                }
            }
            cur.parts.pop_back();
        }

        Combo &selectAttack(std::vector<Combo> &combos)
        {
            int len = combos.size();
            std::random_device dev;
            std::default_random_engine engine(dev());
            int i = engine() % len;
            return combos[i];
        }

        void collectDefense(std::vector<Card> &cards, std::vector<Combo> &combos,
                            int damage, Combo &cur, int i)
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
            for (int j = i + 1; j < cards.size(); ++j)
            {
                collectAttack(cards, combos, false, cur, j);
            }
            cur.parts.pop_back();
        }

        Combo &selectDefense(std::vector<Combo> &combos)
        {
            int len = combos.size();
            std::random_device dev;
            std::default_random_engine engine(dev());
            int i = engine() % len;
            return combos[i];
        }

    } /* namespace dfsel */
} /* namespace regi */
