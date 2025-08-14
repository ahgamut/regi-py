#include <dfsel.h>
#include <random>

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
        (void)g;
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
        (void)g;
        // we only enter this if it is actually possible to block
        std::vector<Combo> combos;
        Combo base;
        for (std::int32_t i = 0; i < player.cards.size(); ++i)
        {
            collectDefense(player.cards, combos, damage, base, i);
        }
        return assign(result, combos);
    }

    void DamageStrategy::collectAttack(const std::vector<Card> &cards,
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

    void DamageStrategy::collectDefense(const std::vector<Card> &cards,
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

    std::int32_t DamageStrategy::calcDamage(const Combo &cur, const Enemy &enemy,
                                        const GameState &g)
    {
        std::uint32_t epow = getPower(enemy) & CLUBS_DOUBLE;
        for (const auto &combo : g.usedPile)
        {
            if ((combo.getPowers() & JOKER_NERF) != 0) { epow = 0; }
        }
        std::int32_t dmg = 0;
        dmg += cur.getBaseDamage();
        bool dbl = ((cur.getPowers() & CLUBS_DOUBLE) & (~epow)) != 0;
        if (dbl) { dmg += cur.getBaseDamage(); }
        return dmg;
    }

    std::int32_t DamageStrategy::provideAttack(Combo &result, const Player &player,
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
        if (combos.size() == 0) { return 0; }
        if (g.enemyPile.empty()) { return 0; }
        const Enemy &enemy = g.enemyPile.front();
        for (auto &cc : combos)
        {
            cc.loadDetails();
        }

        // pick highest-damage combo if cannot kill
        // or pick lowest-killing combo
        std::int32_t dmg = calcDamage(combos[0], enemy, g);
        std::size_t pick = 0;
        std::int32_t tempDmg = 0;
        for (std::size_t i = 1; i < combos.size(); ++i)
        {
            tempDmg = calcDamage(combos[i], enemy, g);
            if (tempDmg >= enemy.hp)
            {
                if (dmg < enemy.hp)
                {
                    dmg = tempDmg;
                    pick = i;
                }
                else if (dmg > tempDmg)
                {
                    dmg = tempDmg;
                    pick = i;
                }
            }
            else
            {
                if (tempDmg > dmg)
                {
                    dmg = tempDmg;
                    pick = i;
                }
            }
        }
        result = combos[pick];
        return 1;
    }

    std::int32_t DamageStrategy::provideDefense(Combo &result, const Player &player,
                                            std::int32_t damage, const GameState &g)
    {
        (void)g;
        // we only enter this if it is actually possible to block
        std::vector<Combo> combos;
        Combo base;
        for (std::int32_t i = 0; i < player.cards.size(); ++i)
        {
            collectDefense(player.cards, combos, damage, base, i);
        }
        if (combos.size() == 0) { return 0; }
        // pick lowest block combo that works
        std::int32_t blk = 1000;
        std::size_t pick = 0;
        std::int32_t tempBlk = 0;
        for (std::size_t i = 0; i < combos.size(); ++i)
        {
            tempBlk = combos[i].getBaseDefense();
            if (tempBlk < blk)
            {
                blk = tempBlk;
                pick = i;
            }
        }
        result = combos[pick];
        return 1;
    }

} /* namespace regi */
