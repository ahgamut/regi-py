#ifndef LOCATION_H
#define LOCATION_H
#include <card.h>
#include <phaseinfo.h>
#include <regi.h>
#include <vector>
#include <utility>
#include <memory>

namespace regi
{
    enum LocationStatus : i32
    {
        NOT_IN_GAME = 0,
        WITH_PLAYER_1 = 1,
        WITH_PLAYER_2 = 2,
        WITH_PLAYER_3 = 3,
        WITH_PLAYER_4 = 4,
        IN_DRAW_PILE = 5,
        IN_DISCARD_PILE = 6,
        IN_USED_PILE = 7,
        IN_ENEMY_PILE = 8,
        MAX_LOCATIONS = 9
    };

    class LocationInfo
    {
       private:
        float *data;
        i32 numJokers;
        i32 numPlayers;
        bool valid;

       public:
        static constexpr i32 rows = MAX_CARDS_IN_GAME;
        static constexpr i32 cols = MAX_LOCATIONS;
        //
        LocationInfo();
        ~LocationInfo();
        //
        void set(i32 i, i32 j, float v = 1.0) { this->data[i * cols + j] = v; }
        float get(i32 i, i32 j) const { return this->data[i * cols + j]; }
        float rowSum(i32 i) const
        {
            float sum = 0;
            for (int j = 0; j < cols; ++j) { sum += this->data[i * cols + j]; }
            return sum;
        }

        void setCard(const Card &, LocationStatus);
        void setProbs(i32, float *);
        void setYield();
        void setJokers();
        void setCards(const std::vector<Card> &, LocationStatus j);
        void setCards(const std::vector<Enemy> &, LocationStatus j);
        void setSurroundings(const std::vector<Card> &,  //
                             const std::vector<Card> &,  //
                             const std::vector<Enemy> &, const std::vector<Combo> &);
        void validate();
        //
        std::vector<std::pair<Card, LocationStatus>> pairwise() const;
        bool validYield() const;
        bool validJokers() const;
        bool getValid() const { return valid; };
        i32 getNumJokers() const { return numJokers; };
        i32 getNumPlayers() const { return numPlayers; };
        float *getData() const { return data; };
        bool nextOK(const LocationInfo &) const;
        //
        static std::shared_ptr<LocationInfo> fromPhaseInfo(const PhaseInfo &);
        static std::shared_ptr<LocationInfo> fromGameState(const GameState &);
        static std::shared_ptr<LocationInfo> fromActivePlayer(const PhaseInfo &, i32);
    };
} /* namespace regi */

#endif
