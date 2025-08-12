#ifndef REGI_H
#define REGI_H
#include <card.h>
#include <combo.h>
#include <enemy.h>
#include <player.h>
#include <utils.h>

namespace regi
{
    struct BaseLog;
    struct GameState
    {
       private:
        BaseLog &log;
        void initPlayers();
        void initDraw();
        void initEnemy();

       public:
        std::int32_t pastYieldsInARow;
        std::int32_t currentRound;
        bool gameRunning;
        Player players[2];
        std::vector<Card> drawPile;    /* cards that can be drawn */
        std::vector<Enemy> enemyPile;  /* enemies still left to KO */
        std::vector<Card> discardPile; /* cards used up to KO enemies */
        std::vector<Combo> usedPile;   /* combos used on current enemy */

        /* methods */
        GameState(BaseLog &l) : log(l) {};
        void init();
        void setup();
        void logAttack(const Player &, const Enemy &, const Combo &,
                       const std::int32_t);
        void logDefend(const Player &, const Combo &, const std::int32_t);
        void logFailBlock(const Player &, const std::int32_t, const std::int32_t);
        void logDrawOne(const Player &);
        void logReplenish(std::int32_t);
        void logState();
        void logDebug();

        void startLoop();
        void oneTurn(Player &);
        void gameOver();
        void postGameResult();

        void playerDraws(Player &, int);
        std::int32_t playerDrawsOne(Player &);
        void refreshDraws(int, int);
        void refreshDiscards(int);

        void selectAttack(Player &, bool);
        std::int32_t calcDamage(Enemy &);
        void attackPhase(Player &, Enemy &);
        void postAttackEffects(Player &, Enemy &);
        std::int32_t enemyDead();

        std::int32_t calcBlock(Enemy &);
        void selectDefense(Player &, int);
        void defensePhase(Player &, Enemy &);
        //
        friend struct BaseLog;
    };

    struct BaseLog
    {
       public:
        virtual void attack(const Player &, const Enemy &, const Combo &,
                            const std::int32_t) = 0;
        virtual void defend(const Player &, const Combo &, const std::int32_t) = 0;
        virtual void failBlock(const Player &, const std::int32_t,
                               const std::int32_t) = 0;
        virtual void drawOne(const Player &) = 0;
        virtual void replenish(const std::int32_t) = 0;
        virtual void state(const GameState &) = 0;
        virtual void debug(const GameState &) = 0;
    };

} /* namespace regi */

#endif
