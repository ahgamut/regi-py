#ifndef PLAYER_H
#define PLAYER_H
#include <card.h>

#include <vector>

namespace regi
{
    struct Player
    {
        static constexpr std::uint32_t HAND_SIZE = 6;

       public:
        int id;
        bool alive;
        std::vector<Card> cards;
        bool full() { return cards.size() == HAND_SIZE; }
    };
    std::ostream& operator<<(std::ostream& os, Player& p);
} /* namespace regi */

#endif
