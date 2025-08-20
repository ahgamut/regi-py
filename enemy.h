#ifndef ENEMY_H
#define ENEMY_H
#include <card.h>

namespace regi
{
    struct Enemy : public Card
    {
       public:
        i32 hp;
        Enemy(Entry ee, Suit ss);
    };
} /* namespace regi */

#endif
