#ifndef COMBO_H
#define COMBO_H
#include <card.h>
#include <vector>

namespace regi
{

    struct Combo
    {
       private:
        std::int32_t baseDmg;
        std::uint32_t powers;

       public:
        std::vector<Card> parts;
        Combo();
        std::int32_t valid(bool);
        void loadDetails();
        std::uint32_t getPowers();
        std::int32_t getBaseDamage();
        std::int32_t getBaseDefense();
    };

} /* namespace regi */

#endif
