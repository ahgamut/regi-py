#include <dfsel.h>
#include <random>
#include <algorithm>

namespace regi
{
    void calcAttackMoves(const std::vector<Card> &cards, std::vector<Combo> &combos,
                         bool yieldAllowed, Combo &cur, i32 i)
    {
        // cards are assumed ordered
        if (i >= cards.size()) { return; }
        u32 tmp = cur.getBitrep();
        // try adding cards[i] to the combo
        cur.parts.push_back(Card(cards[i].entry(), cards[i].suit()));
        cur.setBitrep(tmp | (1 << i));
        //
        if (cur.valid(yieldAllowed))
        {
            // if combo is valid, accumulate,
            // and try extending the combo
            combos.push_back(cur);
            for (i32 j = i + 1; j < cards.size(); ++j)
            {
                calcAttackMoves(cards, combos, yieldAllowed, cur, j);
            }
        }
        //
        cur.setBitrep(tmp);
        cur.parts.pop_back();
    }

    i32 Strategy::provideAttack(Combo &result, const Player &player, bool yieldAllowed,
                                const GameState &g)
    {
        (void)g;
        std::vector<Card> orderedCards;
        std::vector<Combo> combos;
        Combo base;
        if (yieldAllowed && base.parts.size() == 0)
        {
            combos.push_back(base);
            yieldAllowed = false;
        }
        for (i32 i = 0; i < player.cards.size(); ++i)
        {
            orderedCards.push_back(player.cards[i]);
        }
        //
        std::sort(orderedCards.begin(), orderedCards.end());
        for (i32 i = 0; i < player.cards.size(); ++i)
        {
            calcAttackMoves(orderedCards, combos, yieldAllowed, base, i);
        }
        if (combos.size() == 0) { return -1; }
        if (g.enemyPile.empty()) { return -1; }
        for (auto &cc : combos) { cc.loadDetails(); }

        i32 ind = getAttackIndex(combos, player, yieldAllowed, g);
        if (ind < 0 || ind >= combos.size()) { return -1; }
        else
        {
            result = combos[ind];
            return 0;
        }
    }

    void calcDefenseMoves(const std::vector<Card> &cards, std::vector<Combo> &combos,
                          i32 damage, Combo &cur, i32 i)
    {
        // cards are assumed ordered
        if (i >= cards.size()) { return; }
        // try adding cards[i] to the combo
        u32 tmp = cur.getBitrep();
        cur.parts.push_back(Card(cards[i].entry(), cards[i].suit()));
        cur.setBitrep(tmp | (1 << i));
        //
        if (cur.getBaseDefense() >= damage)
        {
            // if combo can defend, accumulate
            combos.push_back(cur);
        }
        // try extending the combo
        for (i32 j = i + 1; j < cards.size(); ++j)
        {
            calcDefenseMoves(cards, combos, damage, cur, j);
        }
        //
        cur.setBitrep(tmp);
        cur.parts.pop_back();
    }

    i32 Strategy::provideDefense(Combo &result, const Player &player, i32 damage,
                                 const GameState &g)
    {
        (void)g;
        // we only enter this if it is actually possible to block
        std::vector<Combo> combos;
        std::vector<Card> orderedCards;
        Combo base;
        for (i32 i = 0; i < player.cards.size(); ++i)
        {
            orderedCards.push_back(player.cards[i]);
        }
        //
        std::sort(orderedCards.begin(), orderedCards.end());
        for (i32 i = 0; i < player.cards.size(); ++i)
        {
            calcDefenseMoves(orderedCards, combos, damage, base, i);
        }
        if (combos.size() == 0) { return -1; }

        i32 ind = getDefenseIndex(combos, player, damage, g);
        if (ind < 0 || ind >= combos.size()) { return -1; }
        else
        {
            result = combos[ind];
            return 0;
        }
    }

    i32 selectRandomCombo(const std::vector<Combo> &combos)
    {
        i32 len = combos.size();
        if (len == 0) return -1;
        std::random_device dev;
        std::default_random_engine engine(dev());
        return engine() % len;
    }

    i32 calcDamage(const Combo &cur, const Enemy &enemy, const GameState &g)
    {
        u32 epow = getPower(enemy) & CLUBS_DOUBLE;
        for (const auto &combo : g.usedPile)
        {
            if ((combo.getPowers() & JOKER_NERF) != 0) { epow = 0; }
        }
        i32 dmg = 0;
        dmg += cur.getBaseDamage();
        bool dbl = ((cur.getPowers() & CLUBS_DOUBLE) & (~epow)) != 0;
        if (dbl) { dmg += cur.getBaseDamage(); }
        return dmg;
    }

    i32 RandomStrategy::setup(const Player &player, const GameState &g)
    {
        (void)player;
        (void)g;
        return 0;
    }

    i32 RandomStrategy::getAttackIndex(const std::vector<Combo> &combos,
                                       const Player &player, bool yieldAllowed,
                                       const GameState &g)
    {
        (void)player;
        (void)yieldAllowed;
        (void)g;
        return selectRandomCombo(combos);
    }

    i32 RandomStrategy::getDefenseIndex(const std::vector<Combo> &combos,
                                        const Player &player, i32 damage,
                                        const GameState &g)
    {
        (void)player;
        (void)damage;
        (void)g;
        return selectRandomCombo(combos);
    }

    i32 DamageStrategy::setup(const Player &player, const GameState &g)
    {
        (void)player;
        (void)g;
        return 0;
    }

    i32 DamageStrategy::getAttackIndex(const std::vector<Combo> &combos,
                                       const Player &player, bool yieldAllowed,
                                       const GameState &g)
    {
        (void)yieldAllowed;
        (void)player;

        const Enemy &enemy = g.enemyPile.front();

        // pick highest-damage combo if cannot kill
        // or pick lowest-killing combo
        i32 dmg = calcDamage(combos[0], enemy, g);
        u64 pick = 0;
        i32 tempDmg = 0;
        for (u64 i = 1; i < combos.size(); ++i)
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
        return static_cast<i32>(pick);
    }

    i32 DamageStrategy::getDefenseIndex(const std::vector<Combo> &combos,
                                        const Player &player, i32 damage,
                                        const GameState &g)
    {
        (void)player;
        (void)damage;
        (void)g;
        // we only enter this if it is actually possible to block
        // pick lowest block combo that works
        i32 blk = 1000;
        u64 pick = 0;
        i32 tempBlk = 0;
        for (u64 i = 0; i < combos.size(); ++i)
        {
            tempBlk = combos[i].getBaseDefense();
            if (tempBlk < blk)
            {
                blk = tempBlk;
                pick = i;
            }
        }
        return static_cast<i32>(pick);
    }

} /* namespace regi */
