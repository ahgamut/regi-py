#include <regi.h>
#include <random>

namespace regi
{

    void GameState::initRandom()
    {
        pastYieldsInARow = 0;
        phaseCount = 0;
        initEnemy();
        initDraw();
        initHandSize();
        // starting at a random point in the game
        std::random_device dev;
        std::default_random_engine engine(dev());

        // some enemies have been killed
        i32 maxEnemies = enemyPile.size();
        i32 killedEnemies = engine() % maxEnemies;
        if (killedEnemies != 0)
        {
            for (i32 i = 0; i < killedEnemies; i++)
            {
                Card econ(enemyPile[i].entry(), enemyPile[i].suit());
                drawPile.push_back(econ);
            }
            enemyPile.erase(enemyPile.begin(), enemyPile.begin() + killedEnemies);
            shuffle(drawPile, 0, drawPile.size());
        }
        // each player may have 0 or more cards
        for (i32 i = 0; i < totalPlayers(); ++i)
        {
            players[i].alive = true;
            players[i].id = i;
            players[i].cards.clear();
            i32 psize = engine() % (handSize + 1);
            playerDraws(players[i], psize);
            if (psize != 0)
            {
                drawPile.erase(drawPile.begin(), drawPile.begin() + psize);
            }
        }

        // some cards are in the discard pile
        i32 numCardsDiscarded = engine() % drawPile.size();
        if (numCardsDiscarded != 0)
        {
            for (i32 i = 0; i < numCardsDiscarded; i++)
            {
                discardPile.push_back(drawPile[i]);
            }
            drawPile.erase(drawPile.begin(), drawPile.begin() + numCardsDiscarded);
            shuffle(discardPile, 0, discardPile.size());
        }

        // no cards are in the used pile
        log.startgame(*this);
    }
} /* namespace regi */
