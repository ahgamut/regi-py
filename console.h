#ifndef LOGGER_H
#define LOGGER_H
#include <regi.h>

namespace regi
{
    class ConsoleLog : public BaseLog
    {
       public:
        void attack(const Player &, const Enemy &, const Combo &, const std::int32_t);
        void enemyKill(const Enemy &, const GameState &);
        void defend(const Player &, const Combo &, const std::int32_t);
        void failBlock(const Player &, const std::int32_t, const std::int32_t);
        void drawOne(const Player &);
        void replenish(const std::int32_t);
        void state(const GameState &);
        void debug(const GameState &);
        void endTurn(const GameState &);
        void startgame(const GameState &);
        void endgame(EndGameReason, const GameState &);
        void postgame(const GameState &);
    };
} /* namespace regi */

#endif
