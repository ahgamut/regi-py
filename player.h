#ifndef PLAYER_H
#define PLAYER_H
#include <card.h>

#include <vector>

namespace regi
{
    struct Player
    {
       public:
        const i32 HAND_SIZE;
        i32 id;
        bool alive;
        std::vector<Card> cards;
        bool full() { return cards.size() == HAND_SIZE; }
        Player(i32 hs) : HAND_SIZE(hs) {};
    };
    std::ostream& operator<<(std::ostream& os, const Player& p);
} /* namespace regi */

#endif
