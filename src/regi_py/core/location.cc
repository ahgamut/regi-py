#include <location.h>

namespace regi
{

    static const i32 validFutureLocations[MAX_LOCATIONS][MAX_LOCATIONS] = {
        /* rows indicate this turn, *
         * cols indicate next turn. */
        {1, 0, 0, 0, 0, 0, 0, 0, 0}, /* NOT_IN_GAME */
        {0, 1, 0, 0, 0, 0, 1, 1, 0}, /* WITH_PLAYER_1 */
        {0, 0, 1, 0, 0, 0, 1, 1, 0}, /* WITH_PLAYER_2 */
        {0, 0, 0, 1, 0, 0, 1, 1, 0}, /* WITH_PLAYER_3 */
        {0, 0, 0, 0, 1, 0, 1, 1, 0}, /* WITH_PLAYER_4 */
        {0, 1, 1, 1, 1, 1, 0, 0, 0}, /* IN_DRAW_PILE */
        {0, 0, 0, 0, 0, 1, 1, 0, 0}, /* IN_DISCARD_PILE */
        {0, 0, 0, 0, 0, 0, 1, 1, 0}, /* IN_USED_PILE */
        {0, 0, 0, 0, 0, 1, 1, 0, 1}  /* IN_ENEMY_PILE */
    };

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

    void LocationInfo::validate()
    {
        if (numPlayers < 2 || numPlayers > 4) return;
        for (int i = 0; i < rows; ++i)
        {
            // every card has a location wrt the game
            if (rowSum(i) < 1) return;
            // non-jokers have to be in the game
            if (i > 1 && this->get(i, CardLocation::NOT_IN_GAME) != 0) return;
        }
        // check joker count summary;
        float nj2 = this->get(0, CardLocation::NOT_IN_GAME) +
                    this->get(1, CardLocation::NOT_IN_GAME);
        if (numJokers + nj2 > 2) return;
        // check if number of jokers is valid
        if (numPlayers == 2 && numJokers != 0) return;
        if (numPlayers == 3 && numJokers != 1) return;
        if (numPlayers == 4 && numJokers != 2) return;
        valid = true;
    }

    bool LocationInfo::nextOK(const LocationInfo &next) const
    {
        if (!this->valid) return false;
        if (!next.valid) return false;
        int i, j1, j2;
        for (i = 0; i < rows; ++i)
        {
            for (j1 = 0; j1 < cols; ++j2)
            {
                if (this->get(i, j1) <= 0) continue;
                for (j2 = 0; j2 < cols; ++j2)
                {
                    if (next.get(i, j2) > 0)
                    {
                        if (!validFutureLocations[j1][j2]) return false;
                    }
                }
            }
        }
        //
        return true;
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
        result->validate();
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
        result->validate();
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
