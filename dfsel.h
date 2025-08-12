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
        std::int32_t assign(Combo &, const std::vector<Combo> &);

       public:
        void collectAttack(const std::vector<Card> &, std::vector<Combo> &, bool,
                           Combo &, int);
        void collectDefense(const std::vector<Card> &, std::vector<Combo> &, int,
                            Combo &, int);
        std::int32_t provideAttack(Combo &, const Player &, bool, const GameState &);
        std::int32_t provideDefense(Combo &, const Player &, std::int32_t,
                                    const GameState &);
    };

    struct DamageStrategy : public Strategy
    {
       private:
        std::int32_t calcDamage(const Combo &, const Enemy &, const GameState &);

       public:
        void collectAttack(const std::vector<Card> &, std::vector<Combo> &, bool,
                           Combo &, int);
        void collectDefense(const std::vector<Card> &, std::vector<Combo> &, int,
                            Combo &, int);
        std::int32_t provideAttack(Combo &, const Player &, bool, const GameState &);
        std::int32_t provideDefense(Combo &, const Player &, std::int32_t,
                                    const GameState &);
    };

} /* namespace regi */

#endif
