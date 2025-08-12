#include <logger.h>

namespace regi
{
    void ConsoleLog::attack(const Player &player, const Enemy &enemy, const Combo &cur,
                            const std::int32_t damage)
    {
        std::cout << "Player " << player.id;
        std::cout << " attacking " << enemy;
        std::cout << " with " << cur;
        std::cout << "\nDealt " << damage << " damage, ";
        std::cout << enemy << " hp is now " << enemy.hp;
        if (enemy.hp <= 0) { std::cout << " (KO)"; }
        std::cout << "\n";
    }

    void ConsoleLog::defend(const Player &player, const Combo &cur,
                            const std::int32_t damage)
    {
        std::cout << "Player " << player.id;
        std::cout << " blocks " << damage;
        std::cout << " damage with " << cur;
        std::cout << "\n";
    }

    void ConsoleLog::failBlock(const Player &player, const std::int32_t damage,
                               const std::int32_t maxblock)
    {
        std::cout << "Player " << player.id;
        std::cout << " needs to block " << damage;
        std::cout << " damage, but can only block for " << maxblock;
        std::cout << "\n";
    }

    void ConsoleLog::drawOne(const Player &player)
    {
        std::cout << "Player " << player.id << " drew a card\n";
    }

    void ConsoleLog::replenish(const std::int32_t n)
    {
        if (n > 0)
        {
            std::cout << "added " << n << " cards from discard pile to draw pile\n";
        }
    }

    void ConsoleLog::state(const GameState &g)
    {
        std::cout << "Round: " << g.currentRound << "\n";
        std::cout << "Player 0: " << g.players[0];
        std::cout << "Player 1: " << g.players[1];
        std::cout << "draw pile has " << g.drawPile.size() << " cards \n";
        std::cout << "discard pile has " << g.discardPile.size() << " cards \n";
        std::cout << "used pile has " << g.usedPile.size() << " combos: ";
        for (auto &u : g.usedPile) { std::cout << u; }
        std::cout << "\n";
        if (g.enemyPile.size() != 0)
        {
            const Enemy &e = g.enemyPile.front();
            std::cout << "current enemy: " << e << " with " << e.hp << "HP\n";
        }
        std::cout << "\n";
    }

    void ConsoleLog::debug(const GameState &g)
    {
        std::cout << "Round: " << g.currentRound << "\n";
        std::cout << "Player 0: " << g.players[0];
        std::cout << "Player 1: " << g.players[1];
        std::cout << "draw pile has " << g.drawPile.size() << " cards: " << g.drawPile;
        std::cout << "discard pile has " << g.discardPile.size()
                  << " cards: " << g.discardPile;
        std::cout << "used pile has " << g.usedPile.size() << " combos: ";
        for (auto &u : g.usedPile) { std::cout << u; }
        std::cout << "\n";
        std::cout << "enemies: " << g.enemyPile;
        if (g.enemyPile.size() != 0)
        {
            const Enemy &e = g.enemyPile.front();
            std::cout << "current enemy: " << e << " with " << e.hp << "HP\n";
        }
        std::cout << "\n";
    }

    void ConsoleLog::endTurn(const GameState &g)
    {
        (void)g;
        std::cout << "\n\n";
    }

    void ConsoleLog::endgame(EndGameReason reason, const GameState &g)
    {
        switch (reason)
        {
            case BLOCK_FAILED:
                std::cout << "endgame: someone was unable to block\n";
                break;
            case ATTACK_FAILED:
                std::cout << "endgame: someone was unable to attack (?)\n";
                break;
            case PLAYER_DEAD:
                std::cout << "endgame: someone was not able to play\n";
                break;
            case NO_ENEMIES:
                break;
        }
    }

    void ConsoleLog::postgame(const GameState &g)
    {
        std::cout << "Game Results: \n";
        if (!g.players[0].alive) { std::cout << "LOST! player 0 KO\n"; }
        else if (!g.players[1].alive) { std::cout << "LOST! player 1 KO\n"; }
        else if (g.enemyPile.size() == 0) { std::cout << "WIN!\n"; }
        else { std::cout << "unknown exit\n"; }

        std::cout << "Game lasted " << g.currentRound << " rounds\n";
        if (g.enemyPile.size() != 0)
        {
            std::cout << "Died to " << g.enemyPile.front() << "\n";
        }
    }

} /* namespace regi */
