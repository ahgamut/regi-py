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

    void LocationInfo::setYield()
    {
        /* a yield "card" is not technically in the game */
        this->set(0, static_cast<i32>(LocationStatus::NOT_IN_GAME));
    }

    bool LocationInfo::validYield() const
    {
        return this->get(0, static_cast<i32>(LocationStatus::NOT_IN_GAME)) > 0;
    }

    void LocationInfo::setJokers()
    {
        for (i32 i = 1; i < 3; ++i)
        {
            if (rowSum(i) == 0) { set(i, LocationStatus::NOT_IN_GAME); }
        }
    }

    bool LocationInfo::validJokers() const
    {
        // check joker count summary;
        float nj2 = this->get(1, LocationStatus::NOT_IN_GAME) +
                    this->get(2, LocationStatus::NOT_IN_GAME);
        if (numJokers + nj2 > 2) return false;
        // check if number of jokers is valid
        if (numPlayers == 2 && numJokers != 0) return false;
        if (numPlayers == 3 && numJokers != 1) return false;
        if (numPlayers == 4 && numJokers != 2) return false;
        return true;
    }

    void LocationInfo::setCard(const Card &c, LocationStatus j)
    {
        i32 loc = c.toLocation();
        if (loc < 0 || loc >= MAX_CARDS_IN_GAME) return;
        if (loc == 1)
        {
            if (numJokers >= 2) return;
            loc = loc + numJokers;
            numJokers++;
        }
        this->set(loc, static_cast<i32>(j));
    }

    void LocationInfo::setCards(const std::vector<Card> &pile, LocationStatus j)
    {
        for (auto &c : pile) this->setCard(c, j);
    }

    void LocationInfo::setCards(const std::vector<Enemy> &pile, LocationStatus j)
    {
        for (auto &c : pile) this->setCard(c, j);
    }

    void LocationInfo::setSurroundings(const std::vector<Card> &drawPile,
                                       const std::vector<Card> &discardPile,
                                       const std::vector<Enemy> &enemyPile,
                                       const std::vector<Combo> &usedPile)
    {
        setCards(drawPile, LocationStatus::IN_DRAW_PILE);
        setCards(discardPile, LocationStatus::IN_DISCARD_PILE);
        setCards(enemyPile, LocationStatus::IN_ENEMY_PILE);
        for (auto &q : usedPile) setCards(q.parts, LocationStatus::IN_USED_PILE);
        setYield();
        setJokers();
    }

    void LocationInfo::validate()
    {
        if (numPlayers < 2 || numPlayers > 4) return;
        if (!validJokers()) return;
        if (!validYield()) return;
        for (int i = 0; i < rows; ++i)
        {
            // every card has a location wrt the game
            if (rowSum(i) < 1) return;
            // non-jokers have to be in the game
            if (i > 1 && this->get(i, LocationStatus::NOT_IN_GAME) != 0) return;
        }
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

    std::vector<std::pair<Card, LocationStatus>> LocationInfo::pairwise() const
    {
        i32 i, j;
        float m;
        std::vector<std::pair<Card, LocationStatus>> res;
        res.resize(MAX_CARDS_IN_GAME - 1);
        for (i = 1; i < MAX_CARDS_IN_GAME; ++i)
        {
            m = this->get(i, 0);
            res[i - 1].first.fromLocation(i);
            res[i - 1].second = static_cast<LocationStatus>(0);
            for (j = 1; j < LocationStatus::MAX_LOCATIONS; ++j)
            {
                if (this->get(i, j) > m)
                {
                    m = this->get(i, j);
                    res[i - 1].second = static_cast<LocationStatus>(j);
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
            result->setCards(p.player_cards[i], static_cast<LocationStatus>(i + 1));
        }
        result->setSurroundings(p.drawPile, p.discardPile, p.enemyPile, p.usedPile);
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
            result->setCards(g.players[i].cards, static_cast<LocationStatus>(i + 1));
        }
        result->setSurroundings(g.drawPile, g.discardPile, g.enemyPile, g.usedPile);
        result->validate();
        return result;
    };

    static i32 relativeID(const PhaseInfo &p, i32 activeID, i32 i)
    {
        i32 r = (activeID + i) % p.numPlayers;
        return r;
    }

    static void fillProbs(const PhaseInfo &p, i32 activeID, float *table)
    {
        float outsideCount = 0.0f;
        i32 n;
        //
        table[IN_DRAW_PILE] = static_cast<float>(p.drawPile.size());
        table[IN_DISCARD_PILE] = static_cast<float>(p.discardPile.size());
        switch (p.numPlayers)
        {
            case 4:
                table[WITH_PLAYER_4] = static_cast<float>(
                    p.player_cards[relativeID(p, activeID, 3)].size());
            // fallthrough
            case 3:
                table[WITH_PLAYER_3] = static_cast<float>(
                    p.player_cards[relativeID(p, activeID, 2)].size());
            // fallthrough
            case 2:
                table[WITH_PLAYER_2] = static_cast<float>(
                    p.player_cards[relativeID(p, activeID, 1)].size());
                break;
        }
        //
        for (n = 0; n < MAX_LOCATIONS; ++n) { outsideCount += table[n]; }
        for (n = 0; n < MAX_LOCATIONS; ++n) { table[n] /= outsideCount; }
    }

    void LocationInfo::setProbs(i32 i, float *table)
    {
        i32 j;
        if (rowSum(i) <= 0.0)
        {
            for (j = 0; j < MAX_LOCATIONS; ++j) { set(i, j, table[j]); }
        }
    }

    std::shared_ptr<LocationInfo> LocationInfo::fromActivePlayer(const PhaseInfo &p,
                                                                 i32 activeID)
    {
        i32 i;
        float table[MAX_LOCATIONS] = {0.0};
        //
        std::shared_ptr<LocationInfo> result = std::make_shared<LocationInfo>();
        result->numJokers = 0;
        result->numPlayers = p.numPlayers;

        // for the active player
        // they know their own cards
        result->setCards(p.player_cards[activeID], LocationStatus::WITH_PLAYER_1);
        // they know what cards are in the used pile
        for (auto &q : p.usedPile)
        {
            result->setCards(q.parts, LocationStatus::IN_USED_PILE);
        }
        // they know all alive enemies are in the enemy pile
        if (p.enemyPile.size() > 0)
        {
            result->setCards(p.enemyPile, LocationStatus::IN_ENEMY_PILE);
        }

        // everything else can be anywhere, so
        // assign uniform probabilities for unknown cards
        fillProbs(p, activeID, table);

        // set joker count explicitly
        if (result->numPlayers >= 2)
        {
            result->numJokers = result->numPlayers - 2;
            for (i = 1; i < 1 + result->numJokers; ++i) { result->setProbs(i, table); }
            for (; i < 3; ++i) { result->set(i, 0); }
        }

        for (i = 3; i < MAX_CARDS_IN_GAME; ++i) { result->setProbs(i, table); }
        result->setYield();
        // no validation
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
