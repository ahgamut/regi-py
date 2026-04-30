#include <location.h>

namespace regi
{
    void LocationInfo::setCard(const Card &c, CardLocation j)
    {
        i32 loc = c.toLocation();
        if (loc < 0 || loc >= MAX_CARDS_IN_GAME) return;
        if (loc == 0)
        {
            loc = numJokers;
            numJokers++;
        }
        this->set(loc, static_cast<i32>(j));
    }

    std::vector<std::pair<Card, CardLocation>> LocationInfo::pairwise() const
    {
        i32 i, j;
        float m;
        std::vector<std::pair<Card, CardLocation>> res;
        res.resize(MAX_CARDS_IN_GAME);
        for (i = 0; i < MAX_CARDS_IN_GAME; ++i)
        {
            m = this->get(i, 0);
            res[i].first.fromLocation(i);
            res[i].second = static_cast<CardLocation>(0);
            for (j = 1; j < CardLocation::MAX_LOCATIONS; ++j)
            {
                if (this->get(i, j) > m)
                {
                    m = this->get(i, j);
                    res[i].second = static_cast<CardLocation>(j);
                }
            }
        }
        return res;
    }

    std::shared_ptr<LocationInfo> LocationInfo::fromPhaseInfo(const PhaseInfo &p)
    {
        std::shared_ptr<LocationInfo> result = std::make_shared<LocationInfo>();
        result->numJokers = 0;
        result->numPlayers = p.numPlayers;
        for (i32 i = 0; i < result->numPlayers; ++i)
        {
            for (auto &c : p.player_cards[i])
            {
                result->setCard(c, static_cast<CardLocation>(i + 1));
            }
        }
        for (auto &c : p.drawPile) { result->setCard(c, CardLocation::IN_DRAW_PILE); }
        for (auto &c : p.discardPile)
        {
            result->setCard(c, CardLocation::IN_DISCARD_PILE);
        }
        for (auto &e : p.enemyPile) { result->setCard(e, CardLocation::IN_ENEMY_PILE); }
        for (auto &q : p.usedPile)
        {
            for (auto &c : q.parts) { result->setCard(c, CardLocation::IN_USED_PILE); }
        }
        for (i32 j = result->numJokers; j < 2; ++j)
        {
            result->set(j, CardLocation::NOT_IN_GAME);
        }
        return result;
    };

    std::shared_ptr<LocationInfo> LocationInfo::fromGameState(const GameState &g)
    {
        std::shared_ptr<LocationInfo> result = std::make_shared<LocationInfo>();
        result->numJokers = 0;
        result->numPlayers = g.players.size();
        for (i32 i = 0; i < result->numPlayers; ++i)
        {
            for (auto &c : g.players[i].cards)
            {
                result->setCard(c, static_cast<CardLocation>(i + 1));
            }
        }
        for (auto &c : g.drawPile) { result->setCard(c, CardLocation::IN_DRAW_PILE); }
        for (auto &c : g.discardPile)
        {
            result->setCard(c, CardLocation::IN_DISCARD_PILE);
        }
        for (auto &e : g.enemyPile) { result->setCard(e, CardLocation::IN_ENEMY_PILE); }
        for (auto &q : g.usedPile)
        {
            for (auto &c : q.parts) { result->setCard(c, CardLocation::IN_USED_PILE); }
        }
        for (i32 j = result->numJokers; j < 2; ++j)
        {
            result->set(j, CardLocation::NOT_IN_GAME);
        }
        return result;
    };

    LocationInfo::LocationInfo()
    {
        /* TODO: (ahgamut) new */
        data = new float[rows * cols];
        numJokers = 0;
        numPlayers = 0;
        valid = false;
        for (i32 x = 0; x < rows * cols; ++x) { data[x] = 0.0; }
    };

    LocationInfo::~LocationInfo() { delete[] data; };
} /* namespace regi */
