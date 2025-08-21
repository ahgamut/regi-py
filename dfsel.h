#ifndef DFSEL_H
#define DFSEL_H
#include <card.h>
#include <player.h>
#include <combo.h>
#include <regi.h>
#include <vector>

namespace regi
{
    struct RandomStrategy : public Strategy
    {
       private:
        i32 assign(Combo &, const std::vector<Combo> &);

       public:
        i32 setup(const Player &, const GameState &);
        i32 getAttackIndex(const std::vector<Combo> &, const Player &, bool,
                           const GameState &);
        i32 getDefenseIndex(const std::vector<Combo> &, const Player &, i32,
                            const GameState &);
    };

    struct DamageStrategy : public Strategy
    {
       private:

       public:
        i32 setup(const Player &, const GameState &);
        i32 getAttackIndex(const std::vector<Combo> &, const Player &, bool,
                           const GameState &);
        i32 getDefenseIndex(const std::vector<Combo> &, const Player &, i32,
                            const GameState &);
    };

    i32 selectRandomCombo(const std::vector<Combo> &);
    i32 calcDamage(const Combo &, const Enemy &, const GameState &);

} /* namespace regi */

#endif
