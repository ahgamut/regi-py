#include <combo.h>

namespace regi
{

    Combo::Combo() : powers(0), baseDmg(0) {};

    int Combo::valid()
    {
        if (parts.size() == 0) { return 0; }
        if (parts.size() == 1)
        {
            return 1;
            /* playing any single card is fine */
        }

        /* if playing multiple parts, cannot play JOKER */
        for (auto c : parts)
        {
            if (c.entry() == JOKER) return 0;
        }

        if (parts.size() == 2 && (parts[0].entry() == ACE || parts[1].entry() == ACE))
        {
            /* ACE + Card combo */
            return 1;
        }

        /* numeric combo: all entries are same, but different suits
         * cannot have ACE, sum must be less than or equal to 10 */
        int sum = 0;
        for (int i = 1; i < parts.size(); ++i)
        {
            sum += static_cast<int>(parts[i].entry());
            if (parts[i].entry() == ACE) return 0;
            if (parts[i].entry() != parts[0].entry()) return 0;
        }
        if (sum > 10) { return 0; }
        return 1;
    }

    void Combo::loadDetails()
    {
        if (!this->valid()) return;
        /* calculate only using card strength,
         * actual damage calc needs context */
        int dmg = 0;
        std::uint32_t pow = 0;
        for (auto c : parts)
        {
            dmg += c.strength();
            pow |= getPower(c);
        }
        this->baseDmg = dmg;
        this->powers = pow;
    }

    int Combo::getBaseDamage() { return this->baseDmg; }
    std::uint32_t Combo::getPowers() { return this->powers; }
} /* namespace regi */


