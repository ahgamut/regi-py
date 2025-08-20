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
        void collectAttack(const std::vector<Card> &, std::vector<Combo> &, bool,
                           Combo &, int);
        void collectDefense(const std::vector<Card> &, std::vector<Combo> &, int,
                            Combo &, int);
        i32 provideAttack(Combo &, const Player &, bool, const GameState &);
        i32 provideDefense(Combo &, const Player &, i32,
                                    const GameState &);
    };

    struct DamageStrategy : public Strategy
    {
       private:
        i32 calcDamage(const Combo &, const Enemy &, const GameState &);

       public:
        void collectAttack(const std::vector<Card> &, std::vector<Combo> &, bool,
                           Combo &, int);
        void collectDefense(const std::vector<Card> &, std::vector<Combo> &, int,
                            Combo &, int);
        i32 provideAttack(Combo &, const Player &, bool, const GameState &);
        i32 provideDefense(Combo &, const Player &, i32,
                                    const GameState &);
    };

} /* namespace regi */

#endif
