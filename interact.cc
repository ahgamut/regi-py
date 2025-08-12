#include <regi.h>
#include <dfsel.h>

namespace regi
{
    void GameState::setup()
    {
        // in a server-style setup we'd initialize the connections here
    }

    void GameState::selectDefense(Player &player, int damage)
    {
        // we only enter this if it is actually possible to block
        std::vector<Combo> combos;
        Combo base;
        for (int i = 0; i < player.cards.size(); ++i)
        {
            dfsel::collectDefense(player.cards, combos, damage, base, i);
        }

        Combo &def = dfsel::selectDefense(combos);
        for (int i = 0; i < def.parts.size(); ++i)
        {
            for (auto it = player.cards.begin(); it != player.cards.end(); ++it)
            {
                if (*it == def.parts[i])
                {
                    player.cards.erase(it);
                    break;
                }
            }
        }

        // add to discard pile
        for (Card &c : def.parts) { discardPile.push_back(c); }
    }

    void GameState::selectAttack(Player &player, bool yieldAllowed)
    {
        std::vector<Combo> combos;
        Combo base;
        if (yieldAllowed && base.parts.size() == 0)
        {
            combos.push_back(base);
            yieldAllowed = false;
        }
        for (int i = 0; i < player.cards.size(); ++i)
        {
            dfsel::collectAttack(player.cards, combos, yieldAllowed, base, i);
        }

        Combo &atk = dfsel::selectAttack(combos);
        atk.loadDetails();
        std::vector<int> removes;
        for (int i = 0; i < atk.parts.size(); ++i)
        {
            for (auto it = player.cards.begin(); it != player.cards.end(); ++it)
            {
                if (*it == atk.parts[i])
                {
                    player.cards.erase(it);
                    break;
                }
            }
        }

        // add to used pile
        usedPile.push_back(atk);
    }

} /* namespace regi */
