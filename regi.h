#ifndef REGI_H
#define REGI_H
#include <vector>
#include <card.h>
#include <combo.h>
#include <enemy.h>
#include <player.h>
#include <utils.h>

namespace regi
{

    struct GameState
    {
       private:
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

        void startLoop();
        void oneTurn(Player &);
        void gameOver();
        void postGameResult();

        void playerDraws(Player &, int n);
        int playerDrawsOne(Player &);
        void refreshDraws(int ip, int n);
        void refreshDiscards(int n);

        void selectAttack(Player &);
        int calcDamage(Enemy &);
        void attackPhase(Player &, Enemy &);
        void postAttackEffects(Player &, Enemy &);
        int enemyDead();

        int calcBlock(Enemy &);
        void selectDefense(Player &, int);
        void defensePhase(Player &, Enemy &);
    };

} /* namespace regi */

#endif
