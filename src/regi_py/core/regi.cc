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
        if (totalPlayers() == 3)
        {
            /* 1 joker for 3p */
            drawPile.push_back(Card(JOKER, GLITCH));
        }
        else if (totalPlayers() == 4)
        {
            /* 2 joker for 4p */
            drawPile.push_back(Card(JOKER, GLITCH));
            drawPile.push_back(Card(JOKER, GLITCH));
        }
        else if (totalPlayers() > 4 || totalPlayers() < 2)
        {
            status = GameStatus::ENDED;
            log.endgame(INVALID_START, *this);
            return;
        }
        shuffle(drawPile, 0, drawPile.size());
    }

    void GameState::initHandSize()
    {
        i32 tp = totalPlayers();
        if (tp > 4 || tp < 2)
        {
            status = GameStatus::ENDED;
            log.endgame(INVALID_START, *this);
            return;
        }
        if (tp == 2) { handSize = 7; }
        else if (tp == 3) { handSize = 6; }
        else /* (tp == 4) */ { handSize = 5; }
    }

    void GameState::initPlayers()
    {
        for (i32 i = 0; i < totalPlayers(); ++i)
        {
            players[i].alive = true;
            players[i].id = i;
            players[i].cards.clear();
            playerDraws(players[i], handSize);
        }
    }

    void GameState::init()
    {
        pastYieldsInARow = 0;
        currentRound = 0;
        initEnemy();
        initDraw();
        initHandSize();
        initPlayers();
        status = GameStatus::RUNNING;
        log.startgame(*this);
    }

    void GameState::setup()
    {
        // in a server-style setup we'd initialize the connections here
        for (i32 i = 0; i < totalPlayers(); ++i)
        {
            if (!players[i].alive || players[i].strat.setup(players[i], *this) != 0)
            {
                status = GameStatus::ENDED;
                log.endgame(INVALID_START, *this);
            }
        }
    }

    i32 GameState::addPlayer(Strategy &s)
    {
        if (status != GameStatus::LOADING) { return -1; }
        players.push_back(Player(s));
        return 0;
    }

} /* namespace regi */
