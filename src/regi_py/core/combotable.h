#ifndef COMBOTABLE_H
#define COMBOTABLE_H
#include <card.h>
#include <combo.h>
#include <phaseinfo.h>
#include <regi.h>
#include <vector>
#include <memory>

namespace regi
{
    enum PlayedStatus : i32
    {
        PLAYED_SELF = 0,
        PLAYED_2_AC,       /* (AC, wx), where wx != AC */
        PLAYED_2_AD,       /* (AD, wx), where wx != AD */
        PLAYED_2_AH,       /* (AH, wx), where wx != AH */
        PLAYED_2_AS,       /* (AS, wx), where wx != AS */
        PLAYED_2_2C,       /* (2C, 2x), where x > C */
        PLAYED_2_2D,       /* (2D, 2x), where x > D */
        PLAYED_2_2H,       /* (2H, wx), where wx == 2S */
        PLAYED_2_3C,       /* (3C, 3x), where x > C */
        PLAYED_2_3D,       /* (3D, 3x), where x > D */
        PLAYED_2_3H,       /* (3H, wx), where wx == 3S */
        PLAYED_2_4C,       /* (4C, 4x), where x > C */
        PLAYED_2_4D,       /* (4D, 4x), where x > D */
        PLAYED_2_4H,       /* (4H, wx), where wx == 4S */
        PLAYED_2_5C,       /* (5C, 5x), where x > C */
        PLAYED_2_5D,       /* (5D, 5x), where x > D */
        PLAYED_2_5H,       /* (5H, wx), where wx == 5S */
        PLAYED_3_2C_2D,    /* (2C, 2D, 2x), where x > D */
        PLAYED_3_2H_2S,    /* (2H, 2S, 2x), where x < H */
        PLAYED_3_3C_3D,    /* (3C, 3D, 3x), where x > D */
        PLAYED_3_3H_3S,    /* (3H, 3S, 3x), where x < H */
        PLAYED_4_2C_2D_2H, /* (2C, 2D, 2H, wx), wx == 2S */
        MAX_PLAYED_STATUS
    };

    class ComboTable
    {
       private:
        u32 *data;
        void setAcePairEntry(const Card&, const Card&);
        void setC2Entry(const Combo&);
        void setC3Entry(const Combo&);

       public:
        static constexpr i32 rows = MAX_CARDS_IN_GAME;
        static constexpr i32 cols = MAX_PLAYED_STATUS;
        //
        ComboTable();
        ~ComboTable();
        //
        void set(i32 i, PlayedStatus j)
        {
            this->data[i * cols + static_cast<i32>(j)] = 1U;
        }
        u32 get(i32 i, PlayedStatus j) const
        {
            return this->data[i * cols + static_cast<i32>(j)];
        }
        u32 rowSum(i32 i) const
        {
            u32 sum = 0;
            for (int j = 0; j < cols; ++j) { sum += this->data[i * cols + j]; }
            return sum;
        }

        void setComboEntry(const Combo &);
        void setYieldEntry();
        void setJokerEntry();
        void setAllCardEntries(const Card &);
        void clearAllCardEntries(const Card &);
        void fromUsedPile(const std::vector<Combo> &pile);
        //
        bool fillComboEntry(const Card &, const PlayedStatus, Combo &) const;
        std::vector<Combo> getAsUsedPile() const;
        //
        u32 *getData() const { return data; };
        //
        static std::shared_ptr<ComboTable> fromPhaseInfo(const PhaseInfo &);
        static std::shared_ptr<ComboTable> fromGameState(const GameState &);
        static std::shared_ptr<ComboTable> allViableEntries();
    };
} /* namespace regi */
#endif
