#include <phaseinfo.h>
#include <regi.h>
#include <algorithm>
#include <iterator>

// shoddy, but I'm checking bounds everywhere
#pragma GCC diagnostic ignored "-Wsign-compare"

namespace regi
{
    static constexpr char SEP3 = ',';
    static constexpr char SEP2 = '#';
    static constexpr char SEP1 = '!';

#define EXPECT_SEPARATOR(ss, expected)            \
    {                                             \
        (ss) >> obtained;                         \
        if (obtained != (expected)) return false; \
    }

#define EXPECT_WITHIN(ss, target, lower, upper)                \
    {                                                               \
        (ss) >> (target);                                           \
        if ((target) < (lower) || (target) > (upper)) return false; \
    }

    bool PhaseInfo::loadFromString(std::string str)
    {
        std::istringstream ss(str);
        char obtained;
        i32 pileSize;
        i32 subSize;
        i32 ind;
        i32 hpvalue;
        // metadata
        ss >> gameHasEnded;
        EXPECT_SEPARATOR(ss, SEP2);
        EXPECT_WITHIN(ss, ind, 0, 1);
        currentPhaseIsAttack = (ind == 1);
        EXPECT_SEPARATOR(ss, SEP2);
        EXPECT_WITHIN(ss, numPlayers, 2, 4);
        EXPECT_SEPARATOR(ss, SEP2);
        EXPECT_WITHIN(ss, activePlayerID, 0, numPlayers-1);
        EXPECT_SEPARATOR(ss, SEP2);
        EXPECT_WITHIN(ss, pastYieldsInARow, 0, numPlayers-1);
        EXPECT_SEPARATOR(ss, SEP1);
        // player cards
        player_cards.resize(numPlayers);
        i32 maxHandSize = 9 - numPlayers; // 9 - 2 = 7
        for (i32 i = 0; i < numPlayers; ++i)
        {
            EXPECT_WITHIN(ss, subSize, 0, maxHandSize);
            player_cards[i].resize(subSize);
            EXPECT_SEPARATOR(ss, SEP2);
            for (i32 j = 0; j < subSize; ++j)
            {
                EXPECT_WITHIN(ss, ind, 0, 70);
                if (!player_cards[i][j].fromIndex(ind)) { return false; }
                if ((j + 1) != subSize) { EXPECT_SEPARATOR(ss, SEP3); }
            }
            if ((i + 1) != numPlayers) { EXPECT_SEPARATOR(ss, SEP2); }
        }
        EXPECT_SEPARATOR(ss, SEP1);

        // enemy pile
        EXPECT_WITHIN(ss, pileSize, 0, 12);
        EXPECT_SEPARATOR(ss, SEP1);
        enemyPile.resize(pileSize);
        for (i32 i = 0; i < pileSize; ++i)
        {
            EXPECT_WITHIN(ss, ind, 0, 70);
            if (!enemyPile[i].fromIndex(ind)) { return false; }
            EXPECT_SEPARATOR(ss, SEP3);
            EXPECT_WITHIN(ss, hpvalue, -40, 40);
            enemyPile[i].hp = hpvalue;
            if ((i + 1) != pileSize) { EXPECT_SEPARATOR(ss, SEP2); }
        }
        EXPECT_SEPARATOR(ss, SEP1);

        // draw pile
        EXPECT_WITHIN(ss, pileSize, 0, 54);
        EXPECT_SEPARATOR(ss, SEP1);
        drawPile.resize(pileSize);
        for (i32 i = 0; i < pileSize; ++i)
        {
            EXPECT_WITHIN(ss, ind, 0, 70);
            if (!drawPile[i].fromIndex(ind)) { return false; }
            if ((i + 1) != drawPile.size()) { EXPECT_SEPARATOR(ss, SEP2); }
        }
        EXPECT_SEPARATOR(ss, SEP1);

        // discard pile
        EXPECT_WITHIN(ss, pileSize, 0, 54);
        EXPECT_SEPARATOR(ss, SEP1);
        discardPile.resize(pileSize);
        for (i32 i = 0; i < pileSize; ++i)
        {
            EXPECT_WITHIN(ss, ind, 0, 70);
            if (!discardPile[i].fromIndex(ind)) { return false; }
            if ((i + 1) != discardPile.size()) { EXPECT_SEPARATOR(ss, SEP2); }
        }
        EXPECT_SEPARATOR(ss, SEP1);

        // combos played
        EXPECT_WITHIN(ss, pileSize, 0, 16);
        EXPECT_SEPARATOR(ss, SEP1);
        usedPile.resize(pileSize);
        for (i32 i = 0; i < pileSize; ++i)
        {
            EXPECT_WITHIN(ss, subSize, 1, 4);
            EXPECT_SEPARATOR(ss, SEP2);
            usedPile[i].parts.resize(subSize);
            for (i32 j = 0; j < subSize; ++j)
            {
                EXPECT_WITHIN(ss, ind, 0, 70);
                if (!usedPile[i].parts[j].fromIndex(ind)) { return false; }
                if ((j + 1) != usedPile[i].parts.size()) { EXPECT_SEPARATOR(ss, SEP3); }
            }
            if ((i + 1) != usedPile.size()) { EXPECT_SEPARATOR(ss, SEP2); }
        }
        EXPECT_SEPARATOR(ss, SEP1);
        // done
        return true;
    }

    std::string PhaseInfo::toString() const
    {
        std::ostringstream ss;
        // metadata
        ss << gameHasEnded << SEP2;
        ss << (currentPhaseIsAttack ? 1 : 0) << SEP2;
        ss << numPlayers << SEP2;
        ss << activePlayerID << SEP2;
        ss << pastYieldsInARow;
        ss << SEP1;
        // player cards
        for (i32 i = 0; i < numPlayers; ++i)
        {
            ss << player_cards[i].size() << SEP2;
            for (i32 j = 0; j < player_cards[i].size(); ++j)
            {
                ss << player_cards[i][j].toIndex();
                if ((j + 1) != player_cards[i].size()) { ss << SEP3; }
            }
            if ((i + 1) != numPlayers) { ss << SEP2; }
        }
        ss << SEP1;
        // enemy pile
        ss << enemyPile.size() << SEP1;
        for (i32 i = 0; i < enemyPile.size(); ++i)
        {
            ss << enemyPile[i].toIndex() << SEP3 << enemyPile[i].hp;
            if ((i + 1) != enemyPile.size()) { ss << SEP2; }
        }
        ss << SEP1;
        // draw pile
        ss << drawPile.size() << SEP1;
        for (i32 i = 0; i < drawPile.size(); ++i)
        {
            ss << drawPile[i].toIndex();
            if ((i + 1) != drawPile.size()) { ss << SEP2; }
        }
        ss << SEP1;
        // discard pile
        ss << discardPile.size() << SEP1;
        for (i32 i = 0; i < discardPile.size(); ++i)
        {
            ss << discardPile[i].toIndex();
            if ((i + 1) != discardPile.size()) { ss << SEP2; }
        }
        ss << SEP1;
        // combos played
        ss << usedPile.size() << SEP1;
        for (i32 i = 0; i < usedPile.size(); ++i)
        {
            ss << usedPile[i].parts.size() << SEP2;
            for (i32 j = 0; j < usedPile[i].parts.size(); ++j)
            {
                ss << usedPile[i].parts[j].toIndex();
                if ((j + 1) != usedPile[i].parts.size()) { ss << SEP3; }
            }
            if ((i + 1) != usedPile.size()) { ss << SEP2; }
        }
        ss << SEP1;
        // done
        return ss.str();
    }

    void GameState::initPhaseInfo(const PhaseInfo &info)
    {
        // metadata
        // gameHasEnded is for later analysis
        activePlayerID = info.activePlayerID;
        currentPhaseIsAttack = info.currentPhaseIsAttack;
        pastYieldsInARow = info.pastYieldsInARow;
        phaseCount = 0;
        // enemy pile
        enemyPile.clear();
        std::copy(info.enemyPile.begin(), info.enemyPile.end(),
                  std::back_inserter(enemyPile));
        // draw pile
        drawPile.clear();
        std::copy(info.drawPile.begin(), info.drawPile.end(),
                  std::back_inserter(drawPile));
        // discard pile
        discardPile.clear();
        std::copy(info.discardPile.begin(), info.discardPile.end(),
                  std::back_inserter(discardPile));
        // combos played
        usedPile.clear();
        std::copy(info.usedPile.begin(), info.usedPile.end(),
                  std::back_inserter(usedPile));
        // player cards
        initHandSize();
        if (info.numPlayers != totalPlayers())
        {
            status = GameStatus::ENDED;
            log.endgame(INVALID_START_PLAYER_COUNT, *this);
            return;
        }
        for (i32 i = 0; i < info.numPlayers; ++i)
        {
            players[i].cards.clear();
            players[i].alive = true;
            players[i].id = i;
            std::copy(info.player_cards[i].begin(), info.player_cards[i].end(),
                      std::back_inserter(players[i].cards));
        }
        status = GameStatus::LOADING;
    }

    void GameState::loadPhaseInfoForExport(PhaseInfo &info)
    {
        // metadata
        if (status != GameStatus::ENDED) { info.gameHasEnded = 0; }
        else
        {
            bool allEnemiesDead = true;
            for (const auto &e : enemyPile)
            {
                allEnemiesDead = allEnemiesDead && e.hp <= 0;
            }
            if (allEnemiesDead) { info.gameHasEnded = 1; }
            else { info.gameHasEnded = -1; }
        }
        info.activePlayerID = activePlayerID;
        info.pastYieldsInARow = pastYieldsInARow;
        info.currentPhaseIsAttack = currentPhaseIsAttack;
        // player cards
        info.numPlayers = totalPlayers();
        info.player_cards.resize(players.size());
        for (i32 i = 0; i < totalPlayers(); ++i)
        {
            info.player_cards[i].clear();
            std::copy(players[i].cards.begin(), players[i].cards.end(),
                      std::back_inserter(info.player_cards[i]));
        }
        // enemy pile
        info.enemyPile.clear();
        std::copy(enemyPile.begin(), enemyPile.end(),
                  std::back_inserter(info.enemyPile));
        // draw pile
        info.drawPile.clear();
        std::copy(drawPile.begin(), drawPile.end(), std::back_inserter(info.drawPile));
        // discard pile
        info.discardPile.clear();
        std::copy(discardPile.begin(), discardPile.end(),
                  std::back_inserter(info.discardPile));
        // combos used
        info.usedPile.clear();
        std::copy(usedPile.begin(), usedPile.end(), std::back_inserter(info.usedPile));
    }
} /* namespace regi */
