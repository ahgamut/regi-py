#include <player.h>

namespace regi
{
    std::ostream& operator<<(std::ostream& os, const Player& p)
    {
        os << "(Player " << p.id;
        os << ", alive: " << (p.alive ? "true" : "false") << "): ";
        os << "[";
        for (u64 i = 0; i < p.cards.size(); ++i)
        {
            os << p.cards[i];
            if (i != p.cards.size() - 1) { os << " "; }
        }
        os << "]\n";
        return os;
    }
} /* namespace regi */
