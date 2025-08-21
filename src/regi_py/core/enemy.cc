#include <enemy.h>

namespace regi
{
    Enemy::Enemy(Entry ee, Suit ss) : Card(ee, ss) { this->hp = 2 * this->strength(); }
} /* namespace regi */
