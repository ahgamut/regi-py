#ifndef ENEMY_H
#define ENEMY_H
#include <card.h>

namespace regi
{
    struct Enemy : public Card
    {
       public:
        std::int32_t hp;
        Enemy(Entry ee, Suit ss);
    };
} /* namespace regi */

#endif
