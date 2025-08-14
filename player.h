#ifndef PLAYER_H
#define PLAYER_H
#include <card.h>

#include <vector>

namespace regi
{
    struct Player
    {
       public:
        const std::int32_t HAND_SIZE;
        std::int32_t id;
        bool alive;
        std::vector<Card> cards;
        bool full() { return cards.size() == HAND_SIZE; }
        Player(std::int32_t hs) : HAND_SIZE(hs) {};
    };
    std::ostream& operator<<(std::ostream& os, const Player& p);
} /* namespace regi */

#endif
