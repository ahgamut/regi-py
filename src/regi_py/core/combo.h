#ifndef COMBO_H
#define COMBO_H
#include <card.h>
#include <vector>

namespace regi
{

    struct Combo
    {
       private:
        i32 baseDmg;
        u32 powers;

       public:
        std::vector<Card> parts;
        Combo();
        bool valid(bool);
        void loadDetails();
        u32 getPowers() const;
        i32 getBaseDamage() const;
        i32 getBaseDefense() const;
        friend std::ostream &operator<<(std::ostream &, const Combo &);
    };

} /* namespace regi */

#endif
