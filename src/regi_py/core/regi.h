#ifndef REGI_H
#define REGI_H
#include <card.h>
#include <combo.h>
#include <enemy.h>
#include <player.h>
#include <utils.h>

namespace regi
{
    enum GameStatus
    {
        LOADING,
        RUNNING,
        ENDED
    };

    enum EndGameReason
    {
        INVALID_START,
        NO_ENEMIES,
        BLOCK_FAILED,
        ATTACK_FAILED,
        PLAYER_DEAD,
    };

    struct BaseLog;
    //
    struct GameState
    {
       private:
        BaseLog &log;
        i32 handSize;
        i32 activePlayerID;
        void initHandSize();
        void initPlayers();
        void initDraw();
        void initEnemy();

       public:
        GameStatus status;
        i32 pastYieldsInARow;
        i32 currentRound;
        std::vector<Player> players;
        std::vector<Card> drawPile;    /* cards that can be drawn */
        std::vector<Enemy> enemyPile;  /* enemies still left to KO */
        std::vector<Card> discardPile; /* cards used up to KO enemies */
        std::vector<Combo> usedPile;   /* combos used on current enemy */

        /* methods */
        GameState(BaseLog &l) : log(l)
        {
            status = GameStatus::LOADING;
            activePlayerID = -1;
        };
        i32 addPlayer(Strategy &);
        void init();
        void setup();
        bool gameRunning() { return this->status == GameStatus::RUNNING; }

        i32 getActivePlayer() { return activePlayerID; }
        i32 getHandSize() { return handSize; }
        i32 totalPlayers() { return static_cast<i32>(players.size()); }

        void startLoop();
        void oneTurn(Player &);
        void gameOver(EndGameReason);
        void postGameResult();

        bool canDraw(Player &);
        void playerDraws(Player &, int);
        i32 playerDrawsOne(Player &);
        void refreshDraws(int, int);
        void refreshDiscards(int);

        void selectAttack(Player &, bool);
        i32 calcDamage(Enemy &);
        void attackPhase(Player &, Enemy &);
        void preAttackEffects(Player &, Enemy &);
        i32 enemyDead();

        i32 calcBlock(Enemy &);
        void selectDefense(Player &, int);
        void defensePhase(Player &, Enemy &);
        //
        friend struct BaseLog;
        friend struct Strategy;
    };

    struct BaseLog
    {
       public:
        virtual void attack(const Player &, const Enemy &, const Combo &, const i32,
                            const GameState &) = 0;
        virtual void defend(const Player &, const Combo &, const i32,
                            const GameState &) = 0;
        virtual void failBlock(const Player &, const i32, const i32,
                               const GameState &) = 0;
        virtual void drawOne(const Player &) = 0;
        virtual void replenish(const i32) = 0;
        virtual void enemyKill(const Enemy &, const GameState &) = 0;
        virtual void state(const GameState &) = 0;
        virtual void debug(const GameState &) = 0;
        virtual void startPlayerTurn(const GameState &) = 0;
        virtual void endPlayerTurn(const GameState &) = 0;
        virtual void startgame(const GameState &) = 0;
        virtual void endgame(EndGameReason, const GameState &) = 0;
        virtual void postgame(const GameState &) = 0;
    };

    struct Strategy
    {
       public:
        virtual i32 setup(const Player &, const GameState &) = 0;
        virtual i32 getDefenseIndex(const std::vector<Combo> &, const Player &, i32,
                                    const GameState &) = 0;
        virtual i32 getAttackIndex(const std::vector<Combo> &, const Player &, bool,
                                   const GameState &) = 0;
        //
        void calcAttackMoves(const std::vector<Card> &, std::vector<Combo> &, bool,
                             Combo &, i32);
        i32 provideAttack(Combo &, const Player &, bool, const GameState &);
        //
        void calcDefenseMoves(const std::vector<Card> &, std::vector<Combo> &, i32,
                              Combo &, i32);
        i32 provideDefense(Combo &, const Player &, i32, const GameState &);
    };

} /* namespace regi */

#endif
