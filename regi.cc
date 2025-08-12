#include <regi.h>

std::ostream &operator<<(std::ostream &os, const std::vector<Card> pile)
{
    for (auto c : pile) { os << c << " "; }
    os << "\n";
    return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<regi::Enemy> pile)
{
    for (auto c : pile) { os << c << " "; }
    os << "\n";
    return os;
}

namespace regi
{
    void GameState::initEnemy()
    {
        enemyPile.push_back(Enemy(JACK, CLUBS));
        enemyPile.push_back(Enemy(JACK, DIAMONDS));
        enemyPile.push_back(Enemy(JACK, HEARTS));
        enemyPile.push_back(Enemy(JACK, SPADES));
        enemyPile.push_back(Enemy(QUEEN, CLUBS));
        enemyPile.push_back(Enemy(QUEEN, DIAMONDS));
        enemyPile.push_back(Enemy(QUEEN, HEARTS));
        enemyPile.push_back(Enemy(QUEEN, SPADES));
        enemyPile.push_back(Enemy(KING, CLUBS));
        enemyPile.push_back(Enemy(KING, DIAMONDS));
        enemyPile.push_back(Enemy(KING, HEARTS));
        enemyPile.push_back(Enemy(KING, SPADES));
        shuffle(enemyPile, 0, 4);
        shuffle(enemyPile, 4, 8);
        shuffle(enemyPile, 8, 12);
    }

    void GameState::initDraw()
    {
        drawPile.push_back(Card(ACE, SPADES));      //
        drawPile.push_back(Card(TWO, SPADES));      //
        drawPile.push_back(Card(THREE, SPADES));    //
        drawPile.push_back(Card(FOUR, SPADES));     //
        drawPile.push_back(Card(FIVE, SPADES));     //
        drawPile.push_back(Card(SIX, SPADES));      //
        drawPile.push_back(Card(SEVEN, SPADES));    //
        drawPile.push_back(Card(EIGHT, SPADES));    //
        drawPile.push_back(Card(NINE, SPADES));     //
        drawPile.push_back(Card(TEN, SPADES));      //
                                                    /* new suit */
        drawPile.push_back(Card(ACE, HEARTS));      //
        drawPile.push_back(Card(TWO, HEARTS));      //
        drawPile.push_back(Card(THREE, HEARTS));    //
        drawPile.push_back(Card(FOUR, HEARTS));     //
        drawPile.push_back(Card(FIVE, HEARTS));     //
        drawPile.push_back(Card(SIX, HEARTS));      //
        drawPile.push_back(Card(SEVEN, HEARTS));    //
        drawPile.push_back(Card(EIGHT, HEARTS));    //
        drawPile.push_back(Card(NINE, HEARTS));     //
        drawPile.push_back(Card(TEN, HEARTS));      //
                                                    /* new suit */
        drawPile.push_back(Card(ACE, DIAMONDS));    //
        drawPile.push_back(Card(TWO, DIAMONDS));    //
        drawPile.push_back(Card(THREE, DIAMONDS));  //
        drawPile.push_back(Card(FOUR, DIAMONDS));   //
        drawPile.push_back(Card(FIVE, DIAMONDS));   //
        drawPile.push_back(Card(SIX, DIAMONDS));    //
        drawPile.push_back(Card(SEVEN, DIAMONDS));  //
        drawPile.push_back(Card(EIGHT, DIAMONDS));  //
        drawPile.push_back(Card(NINE, DIAMONDS));   //
        drawPile.push_back(Card(TEN, DIAMONDS));    //
                                                    /* new suit */
        drawPile.push_back(Card(ACE, CLUBS));       //
        drawPile.push_back(Card(TWO, CLUBS));       //
        drawPile.push_back(Card(THREE, CLUBS));     //
        drawPile.push_back(Card(FOUR, CLUBS));      //
        drawPile.push_back(Card(FIVE, CLUBS));      //
        drawPile.push_back(Card(SIX, CLUBS));       //
        drawPile.push_back(Card(SEVEN, CLUBS));     //
        drawPile.push_back(Card(EIGHT, CLUBS));     //
        drawPile.push_back(Card(NINE, CLUBS));      //
        drawPile.push_back(Card(TEN, CLUBS));       //
                                                    /* 1 joker for 2p */
        drawPile.push_back(Card(JOKER, GLITCH));
        shuffle(drawPile, 0, 41);
    }

    void GameState::initPlayers()
    {
        players[0].alive = true;
        players[0].id = 0;
        playerDraws(players[0], Player::HAND_SIZE);
        players[1].alive = true;
        players[1].id = 1;
        playerDraws(players[1], Player::HAND_SIZE);
    }

    void GameState::init()
    {
        pastYieldsInARow = 0;
        currentRound = 0;
        initEnemy();
        initDraw();
        initPlayers();
        gameRunning = true;
    }

    /* logging */
    void GameState::logAttack(const Player &player, const Enemy &enemy,
                              const Combo &cur, const std::int32_t damage)
    {
        std::cout << "Player " << player.id;
        std::cout << " attacking " << enemy;
        std::cout << " with " << cur;
        std::cout << "\nDealt " << damage << " damage, ";
        std::cout << enemy << " hp is now " << enemy.hp;
        if (enemy.hp <= 0) { std::cout << " (KO)"; }
        std::cout << "\n";
    }

    void GameState::logDefend(const Player &player, const Combo &cur,
                              const std::int32_t damage)
    {
        std::cout << "Player " << player.id;
        std::cout << " attacked for " << damage;
        std::cout << ", blocked with " << cur;
        std::cout << "\n";
    }

    void GameState::logFailBlock(const Player &player, const std::int32_t damage,
                                 const std::int32_t maxblock)
    {
        std::cout << "Player " << player.id;
        std::cout << " attacked for " << damage;
        std::cout << ", but can only block for " << maxblock;
        std::cout << "\n";
    }

    void GameState::logDrawOne(const Player &player)
    {
        std::cout << "Player " << player.id << " drew a card\n";
    }

    void GameState::logReplenish(std::int32_t n)
    {
        std::cout << "added " << n << " cards from discard pile to draw pile\n";
    }

    void GameState::logState()
    {
        std::cout << "Round: " << currentRound << "\n";
        std::cout << "Player 0: " << players[0];
        std::cout << "Player 1: " << players[1];
        std::cout << "draw pile has " << drawPile.size() << " cards \n";
        std::cout << "discard pile has " << discardPile.size() << " cards \n";
        std::cout << "used pile has " << usedPile.size() << " combos: ";
        for (auto &u : usedPile) { std::cout << u; }
        std::cout << "\n";
        if (enemyPile.size() != 0)
        {
            Enemy &e = enemyPile.front();
            std::cout << "current enemy: " << e << " with " << e.hp << "HP\n";
        }
        std::cout << "\n";
    }

    void GameState::logDebug()
    {
        std::cout << "Round: " << currentRound << "\n";
        std::cout << "Player 0: " << players[0];
        std::cout << "Player 1: " << players[1];
        std::cout << "draw pile has " << drawPile.size() << " cards: " << drawPile;
        std::cout << "discard pile has " << discardPile.size()
                  << " cards: " << discardPile;
        std::cout << "used pile has " << usedPile.size() << " combos: ";
        for (auto &u : usedPile) { std::cout << u; }
        std::cout << "\n";
        std::cout << "enemies: " << enemyPile;
        if (enemyPile.size() != 0)
        {
            Enemy &e = enemyPile.front();
            std::cout << "current enemy: " << e << " with " << e.hp << "HP\n";
        }
        std::cout << "\n";
    }

} /* namespace regi */
