#ifndef COMBO_H
#define COMBO_H
#include <card.h>
#include <vector>

namespace regi
{

    struct Combo
    {
       private:
        int baseDmg;
        std::uint32_t powers;
        void loadDetails();

       public:
        Combo();
        int valid();
        std::vector<Card> parts;
        std::uint32_t getPowers();
        int getBaseDamage();
    };

} /* namespace regi */

std::ostream& operator<<(std::ostream& os, regi::Combo& c);
#endif
