#include <player.h>

namespace regi
{
    std::ostream& operator<<(std::ostream& os, Player& p)
    {
        for (auto c : p.cards) { os << c << " "; }
        os << "\n";
        return os;
    }
} /* namespace regi */
