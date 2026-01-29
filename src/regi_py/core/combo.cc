#include <combo.h>

namespace regi
{

    Combo::Combo() : baseDmg(0), powers(0), bitrep(0) {};

    bool Combo::valid(bool yieldAllowed)
    {
        if (parts.size() == 0)
        {
            // yielding depends on context
            return yieldAllowed;
        }
        if (parts.size() == 1)
        {
            /* playing any single card is fine */
            return true;
        }

        /* if playing multiple parts, cannot play JOKER */
        for (auto c : parts)
        {
            if (c.entry() == JOKER) return false;
        }

        if (parts.size() == 2 && (parts[0].entry() == ACE || parts[1].entry() == ACE))
        {
            /* ACE + Card combo */
            return true;
        }

        /* numeric combo: all entries are same, but different suits
         * cannot have ACE, sum must be less than or equal to 10 */
        i32 sum = 0;
        sum += static_cast<i32>(parts[0].entry());
        for (i32 i = 1; i < parts.size(); ++i)
        {
            sum += static_cast<i32>(parts[i].entry());
            if (parts[i].entry() == ACE) return false;
            if (parts[i].entry() != parts[0].entry()) return false;
        }
        return sum <= 10;
    }

    void Combo::loadDetails()
    {
        /* calculate only using card strength,
         * actual damage calc needs context */
        i32 dmg = 0;
        u32 pow = 0;
        for (auto c : parts)
        {
            dmg += c.strength();
            pow |= getPower(c);
        }
        this->baseDmg = dmg;
        this->powers = pow;
    }

    i32 Combo::getBaseDefense() const
    {
        /* combo does not need to be valid
         * for calculating defense */
        i32 blk = 0;
        for (auto c : parts) { blk += c.strength(); }
        return blk;
    }

    i32 Combo::getBaseDamage() const { return this->baseDmg; }
    u32 Combo::getPowers() const { return this->powers; }

    std::ostream &operator<<(std::ostream &os, const regi::Combo &combo)
    {
        if (combo.parts.empty()) { os << "(yield)"; }
        else
        {
            os << "(";
            for (u64 i = 0; i < combo.parts.size(); ++i)
            {
                os << combo.parts[i];
                if (i != combo.parts.size() - 1) { os << " "; }
            }
            os << ")";
        }
        return os;
    }

    u32 Combo::getBitrep() const { return this->bitrep; }
    void Combo::setBitrep(u32 br) {
        this->bitrep = br;
    }
} /* namespace regi */

