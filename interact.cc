#include <regi.h>
#include <dfsel.h>

namespace regi
{
    void GameState::setup()
    {
        // in a server-style setup we'd initialize the connections here
    }

    void GameState::selectDefense(Player &player, std::int32_t damage)
    {
        Combo def;

        if (strat.provideDefense(def, player, damage, *this) == 0)
        {
            player.alive = false;
            gameOver();
            return;
        }
        for (std::int32_t i = 0; i < def.parts.size(); ++i)
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
        log.defend(player, def, damage);
        // add to discard pile
        for (Card &c : def.parts) { discardPile.push_back(c); }
    }

    void GameState::selectAttack(Player &player, bool yieldAllowed)
    {
        Combo atk;

        if (strat.provideAttack(atk, player, yieldAllowed, *this) == 0)
        {
            player.alive = false;
            gameOver();
            return;
        }
        atk.loadDetails();
        std::vector<int> removes;
        for (std::int32_t i = 0; i < atk.parts.size(); ++i)
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
