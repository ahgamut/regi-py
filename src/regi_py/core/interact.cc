#include <regi.h>
#include <dfsel.h>

namespace regi
{
    void GameState::selectDefense(Player &player, i32 damage)
    {
        Combo def;
        Strategy &strat = player.strat;

        if (strat.provideDefense(def, player, damage, *this) < 0)
        {
            player.alive = false;
            gameOver(BLOCK_FAILED);
            return;
        }
        for (i32 i = 0; i < def.parts.size(); ++i)
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
        log.defend(player, def, damage, *this);
        // add to discard pile
        for (Card &c : def.parts) { discardPile.push_back(c); }
    }

    void GameState::selectAttack(Player &player, bool yieldAllowed)
    {
        Combo atk;
        Strategy &strat = player.strat;

        if (strat.provideAttack(atk, player, yieldAllowed, *this) < 0)
        {
            player.alive = false;
            gameOver(ATTACK_FAILED);
            return;
        }
        atk.loadDetails();
        std::vector<int> removes;
        for (i32 i = 0; i < atk.parts.size(); ++i)
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

    void GameState::selectRedirect(Player &player)
    {
        Combo atk;
        Strategy &strat = player.strat;

        i32 nextPlayerID = strat.provideRedirect(player, *this);
        if (nextPlayerID < 0)
        {
            player.alive = false;
            gameOver(REDIRECT_FAILED);
            return;
        }

        log.redirect(player, nextPlayerID, *this);
        currentPhaseIsAttack = true;
        activePlayerID = nextPlayerID;
    }

} /* namespace regi */
