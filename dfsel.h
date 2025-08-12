#ifndef DFSEL_H
#define DFSEL_H
#include <card.h>
#include <player.h>
#include <combo.h>
#include <vector>
#include <random>

namespace regi
{
    namespace dfsel
    {

        void collectAttack(std::vector<Card> &, std::vector<Combo> &, bool, Combo &,
                           int);
        Combo &selectAttack(std::vector<Combo> &);

        void collectDefense(std::vector<Card> &, std::vector<Combo> &, int, Combo &,
                            int);
        Combo &selectDefense(std::vector<Combo> &);
    } /* namespace dfsel */
} /* namespace regi */

#endif
