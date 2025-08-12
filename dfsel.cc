#include <dfsel.h>

namespace regi
{
    void RandomStrategy::collectAttack(const std::vector<Card> &cards,
                                       std::vector<Combo> &combos, bool yieldAllowed,
                                       Combo &cur, std::int32_t i)
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

    void RandomStrategy::collectDefense(const std::vector<Card> &cards,
                                        std::vector<Combo> &combos, std::int32_t damage,
                                        Combo &cur, std::int32_t i)
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
            collectDefense(cards, combos, damage, cur, j);
        }
        cur.parts.pop_back();
    }

    std::int32_t RandomStrategy::assign(Combo &result, const std::vector<Combo> &combos)
    {
        std::int32_t len = combos.size();
        if (len == 0) return 0;
        std::random_device dev;
        std::default_random_engine engine(dev());
        std::int32_t i = engine() % len;
        result = combos[i];
        return 1;
    }

    std::int32_t RandomStrategy::provideAttack(Combo &result, const Player &player,
                                               bool yieldAllowed, const GameState &g)
    {
        std::vector<Combo> combos;
        Combo base;
        if (yieldAllowed && base.parts.size() == 0)
        {
            combos.push_back(base);
            yieldAllowed = false;
        }
        for (std::int32_t i = 0; i < player.cards.size(); ++i)
        {
            collectAttack(player.cards, combos, yieldAllowed, base, i);
        }
        return assign(result, combos);
    }

    std::int32_t RandomStrategy::provideDefense(Combo &result, const Player &player,
                                                std::int32_t damage, const GameState &g)
    {
        // we only enter this if it is actually possible to block
        std::vector<Combo> combos;
        Combo base;
        for (std::int32_t i = 0; i < player.cards.size(); ++i)
        {
            collectDefense(player.cards, combos, damage, base, i);
        }
        return assign(result, combos);
    }

} /* namespace regi */
