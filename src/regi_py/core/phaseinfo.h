#ifndef PHASEINFO_H
#define PHASEINFO_H
#include <card.h>
#include <combo.h>
#include <enemy.h>
#include <player.h>
#include <utils.h>
#include <string>
#include <sstream>

namespace regi
{
    /* PhaseInfo contains no info about players, only card info.
     * with this info, we should be able to resume paused games,
     * as long as the correct number of players are added in. */
    struct PhaseInfo
    {
        i32 gameHasEnded;
        bool currentPhaseIsAttack;
        i32 numPlayers;
        i32 activePlayerID;
        i32 pastYieldsInARow;
        // cards
        std::vector<std::vector<Card>> player_cards;
        std::vector<Card> drawPile;    /* cards that can be drawn */
        std::vector<Enemy> enemyPile;  /* enemies still left to KO */
        std::vector<Card> discardPile; /* cards used up to KO enemies */
        std::vector<Combo> usedPile;   /* combos used on current enemy */

        PhaseInfo() {};
        bool loadFromString(std::string);
        std::string toString() const;
    };
} /* namespace regi */
#endif
