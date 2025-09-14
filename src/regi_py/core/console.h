#ifndef LOGGER_H
#define LOGGER_H
#include <regi.h>

namespace regi
{
    class ConsoleLog : public BaseLog
    {
       public:
        void attack(const Player &, const Enemy &, const Combo &, const i32, const GameState &);
        void enemyKill(const Enemy &, const GameState &);
        void defend(const Player &, const Combo &, const i32, const GameState &);
        void failBlock(const Player &, const i32, const i32, const GameState &);
        void drawOne(const Player &);
        void replenish(const i32);
        void state(const GameState &);
        void debug(const GameState &);
        void startPlayerTurn(const GameState &);
        void endPlayerTurn(const GameState &);
        void startgame(const GameState &);
        void endgame(EndGameReason, const GameState &);
        void postgame(const GameState &);
    };
} /* namespace regi */

#endif
