#ifndef REGI_H
#define REGI_H
#include <card.h>
#include <combo.h>
#include <enemy.h>
#include <player.h>
#include <utils.h>

namespace regi
{
    enum Event {
        ATTACK,
        DEFEND,
        DRAW,
        REPLENISH
    };

    struct GameState
    {
       private:
        std::int32_t pastYieldsInARow;
        bool gameRunning;
        void initPlayers();
        void initDraw();
        void initEnemy();

       public:
        Player players[2];
        std::vector<Card> drawPile;    /* cards that can be drawn */
        std::vector<Enemy> enemyPile;  /* enemies still left to KO */
        std::vector<Card> discardPile; /* cards used up to KO enemies */
        std::vector<Combo> usedPile;   /* combos used on current enemy */

        /* methods */
        void init();
        void setup();
        void logEvent(const Event ev, Player &);
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
    };

} /* namespace regi */

#endif
