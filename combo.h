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

       public:
        std::vector<Card> parts;
        Combo();
        int valid(bool);
        void loadDetails();
        std::uint32_t getPowers();
        int getBaseDamage();
        int getBaseDefense();
    };

} /* namespace regi */

#endif
