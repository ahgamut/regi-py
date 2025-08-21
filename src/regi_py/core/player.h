#ifndef PLAYER_H
#define PLAYER_H
#include <card.h>

#include <vector>

namespace regi
{
    struct Strategy;
    //
    struct Player
    {
       public:
        struct Strategy& strat;
        i32 id;
        bool alive;
        std::vector<Card> cards;
        Player(struct Strategy& s) : strat(s)
        {
            id = 0;
            alive = false;
        };
    };
    std::ostream& operator<<(std::ostream& os, const Player& p);
} /* namespace regi */

#endif
