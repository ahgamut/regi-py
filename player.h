#ifndef PLAYER_H
#define PLAYER_H
#include <card.h>

#include <vector>

#ifndef NUM_PLAYERS
#define NUM_PLAYERS 2
#endif

namespace regi
{
    struct Player
    {
       public:
#if NUM_PLAYERS == 2
        static constexpr std::uint32_t HAND_SIZE = 7;
#elif NUM_PLAYERS == 3
        static constexpr std::uint32_t HAND_SIZE = 6;
#elif NUM_PLAYERS == 4
        static constexpr std::uint32_t HAND_SIZE = 5;
#else
    #pragma GCC error "only 2, 3, or 4 players!"
#endif
        std::int32_t id;
        bool alive;
        std::vector<Card> cards;
        bool full() { return cards.size() == HAND_SIZE; }
    };
    std::ostream& operator<<(std::ostream& os, const Player& p);
} /* namespace regi */

#endif
