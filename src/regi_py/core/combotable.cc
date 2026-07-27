#include <combotable.h>
#include <set>

namespace regi
{

    void ComboTable::setYieldEntry() { this->set(LOCATION_YIELD, PLAYED_SELF); }

    bool ComboTable::fillComboEntry(const Card &card, PlayedStatus s, Combo &combo)
    {
        combo.parts.clear();
        bool valid = false;
        i32 offset = 0;
        Card tmp;
        switch (s)
        {
            case PLAYED_2_AC:
            case PLAYED_2_AD:
            case PLAYED_2_AH:
            case PLAYED_2_AS:
                offset = static_cast<i32>(CLUBS) + static_cast<i32>(s) -
                         static_cast<i32>(PLAYED_2_AC);
                if (card.entry() == ACE and card.suit() <= offset) { return false; }
                tmp = Card(ACE, static_cast<Suit>(offset));
                combo.parts.push_back(tmp);
                break;

            case PLAYED_2_2C:
            case PLAYED_2_2D:
            case PLAYED_2_2H:
                if (card.entry() == ACE) return false;
                offset = static_cast<i32>(CLUBS) + static_cast<i32>(s) -
                         static_cast<i32>(PLAYED_2_2C);
                if (card.suit() <= offset) return false;
                tmp = Card(TWO, static_cast<Suit>(offset));
                combo.parts.push_back(tmp);
                break;

            case PLAYED_2_3C:
            case PLAYED_2_3D:
            case PLAYED_2_3H:
                if (card.entry() == ACE) return false;
                offset = static_cast<i32>(CLUBS) + static_cast<i32>(s) -
                         static_cast<i32>(PLAYED_2_3C);
                if (card.suit() <= offset) return false;
                tmp = Card(THREE, static_cast<Suit>(offset));
                combo.parts.push_back(tmp);
                break;

            case PLAYED_2_4C:
            case PLAYED_2_4D:
            case PLAYED_2_4H:
                if (card.entry() == ACE) return false;
                offset = static_cast<i32>(CLUBS) + static_cast<i32>(s) -
                         static_cast<i32>(PLAYED_2_4C);
                if (card.suit() <= offset) return false;
                tmp = Card(FOUR, static_cast<Suit>(offset));
                combo.parts.push_back(tmp);
                break;

            case PLAYED_2_5C:
            case PLAYED_2_5D:
            case PLAYED_2_5H:
                if (card.entry() == ACE) return false;
                offset = static_cast<i32>(CLUBS) + static_cast<i32>(s) -
                         static_cast<i32>(PLAYED_2_5C);
                if (card.suit() <= offset) return false;
                tmp = Card(FIVE, static_cast<Suit>(offset));
                combo.parts.push_back(tmp);
                break;

            case PLAYED_3_2C_2D:
                combo.parts.emplace_back(Card{TWO, CLUBS});
                combo.parts.emplace_back(Card{TWO, DIAMONDS});
                break;

            case PLAYED_3_2H_2S:
                combo.parts.emplace_back(Card{TWO, HEARTS});
                combo.parts.emplace_back(Card{TWO, SPADES});
                break;

            case PLAYED_3_3C_3D:
                combo.parts.emplace_back(Card{THREE, CLUBS});
                combo.parts.emplace_back(Card{THREE, DIAMONDS});
                break;

            case PLAYED_3_3H_3S:
                combo.parts.emplace_back(Card{THREE, HEARTS});
                combo.parts.emplace_back(Card{THREE, SPADES});
                break;

            case PLAYED_4_2C_2D_2H:
                combo.parts.emplace_back(Card{TWO, CLUBS});
                combo.parts.emplace_back(Card{TWO, DIAMONDS});
                combo.parts.emplace_back(Card{TWO, HEARTS});
                break;

            case PLAYED_SELF:
                // adding at end
                break;

            default:
                return false;
        }
        combo.parts.push_back(card);
        valid = combo.valid(true);
        if (!valid) { combo.parts.clear(); }
        /* populate baseDmg/powers/bitrep so every combo built from the table
         * carries its canonical bitwise identity */
        else { combo.loadDetails(); }
        return valid;
    }

    void ComboTable::clearAllCardEntries(const Card &c)
    {
        i32 i = c.toLocation();
        if (i < 1) return;
        for (i32 j = 0; j < cols; ++j) { this->data[i * cols + j] = 0; }
    }

    void ComboTable::setAllCardEntries(const Card &c)
    {
        Combo combo;
        i32 j = 0;
        for (j = 0; j < cols; ++j)
        {
            if (fillComboEntry(c, static_cast<PlayedStatus>(j), combo))
            {
                this->set(c.toLocation(), static_cast<PlayedStatus>(j));
            }
        }
    }

    void ComboTable::setAcePairEntry(const Card &ace, const Card &other)
    {
        i32 card_base = static_cast<i32>(PLAYED_2_AC);
        i32 suit_offset = static_cast<i32>(ace.suit()) - static_cast<i32>(Suit::CLUBS);
        /* status set with respect to the not-ace card */
        this->set(other.toLocation(),
                  static_cast<PlayedStatus>(card_base + suit_offset));
    }

    void ComboTable::setC2Entry(const Combo &combo)
    {
        if (combo.parts[0].entry() == ACE && combo.parts[1].entry() != ACE)
        {
            setAcePairEntry(combo.parts[0], combo.parts[1]);
        }
        else if (combo.parts[1].entry() == ACE && combo.parts[0].entry() != ACE)
        {
            setAcePairEntry(combo.parts[1], combo.parts[0]);
        }
        else
        {
            std::set<Card> cset;
            i32 card_base;
            i32 suit_offset;
            //
            for (auto &c : combo.parts) { cset.insert(c); }
            auto it = cset.begin();
            switch (it->entry())
            {
                case ACE:
                    card_base = PLAYED_2_AC;
                    break;
                case TWO:
                    card_base = PLAYED_2_2C;
                    break;
                case THREE:
                    card_base = PLAYED_2_3C;
                    break;
                case FOUR:
                    card_base = PLAYED_2_4C;
                    break;
                case FIVE:
                    card_base = PLAYED_2_5C;
                    break;
                default:
                    return;
            }
            /* in sorted 2-card combos, the first card identifies status */
            suit_offset = static_cast<i32>(it->suit()) - static_cast<i32>(Suit::CLUBS);
            /* status set with respect to the last card */
            ++it;
            this->set(it->toLocation(),
                      static_cast<PlayedStatus>(card_base + suit_offset));
        }
    }

    void ComboTable::setC3Entry(const Combo &combo)
    {
        std::set<Card> cset;
        i32 card_base;
        i32 suit_offset;
        //
        for (auto &c : combo.parts) { cset.insert(c); }
        auto it = cset.begin();
        switch (it->entry())
        {
            case TWO:
                card_base = PLAYED_3_2C_2D;
                break;
            case THREE:
                card_base = PLAYED_3_3C_3D;
                break;
            default:
                return;
        }
        /* in sorted 3-card combos, the second card identifies status */
        ++it;
        suit_offset = static_cast<i32>(it->suit()) - static_cast<i32>(Suit::DIAMONDS);
        /* status set with respect to the last card */
        ++it;
        this->set(it->toLocation(), static_cast<PlayedStatus>(card_base + suit_offset));
    }

    void ComboTable::setComboEntry(const Combo &combo)
    {
        if (!combo.valid(true)) return;
        u32 numParts = combo.parts.size();
        switch (numParts)
        {
            case 0:
                this->setYieldEntry();
                break;
            case 1:
                /* a single card (including a joker, which now has its own
                 * distinct location) is just PLAYED_SELF at its location */
                this->set(combo.parts[0].toLocation(), PLAYED_SELF);
                break;
            case 2:
                this->setC2Entry(combo);
                break;
            case 3:
                this->setC3Entry(combo);
                break;
            case 4:
                Card two_spade(TWO, SPADES);
                this->set(two_spade.toLocation(), PLAYED_4_2C_2D_2H);
        }
    }

    ComboTable::ComboTable()
    {
        /* TODO: (ahgamut) new */
        data = new u32[rows * cols];
        for (i32 x = 0; x < rows * cols; ++x) { data[x] = 0; }
    };

    ComboTable::~ComboTable() { delete[] data; };

    void ComboTable::fromUsedPile(const std::vector<Combo> &pile)
    {
        for (auto &combo : pile) { setComboEntry(combo); }
    }

    std::vector<Combo> ComboTable::getAsUsedPile() const
    {
        std::vector<Combo> res;
        Combo combo;
        Card card;
        i32 i, j;
        if (this->get(0, PLAYED_SELF)) { res.push_back(combo); }
        for (i = 1; i < rows; ++i)
        {
            if (rowSum(i) == 0) continue;
            card.fromLocation(i);
            for (j = 0; j < cols; ++j)
            {
                if (this->data[i * cols + j] == 0) continue;
                if (!fillComboEntry(card, static_cast<PlayedStatus>(j), combo))
                    continue;
                res.push_back(combo);
            }
        }
        return res;
    }

    std::shared_ptr<ComboTable> ComboTable::fromPhaseInfo(const PhaseInfo &info)
    {
        std::shared_ptr<ComboTable> result = std::make_shared<ComboTable>();
        result->fromUsedPile(info.usedPile);
        return result;
    }

    std::shared_ptr<ComboTable> ComboTable::fromGameState(const GameState &game)
    {
        std::shared_ptr<ComboTable> result = std::make_shared<ComboTable>();
        result->fromUsedPile(game.usedPile);
        return result;
    }

    std::shared_ptr<ComboTable> ComboTable::allViableEntries()
    {
        std::shared_ptr<ComboTable> result = std::make_shared<ComboTable>();
        result->setYieldEntry();  // location 0
        Combo combo;
        Card card;
        i32 i, j;
        for (i = 1; i < result->rows; ++i)
        {
            // resign is not a playable card; yield is handled above (i starts at 1)
            if (i == LOCATION_RESIGN) continue;
            card.fromLocation(i);
            for (j = 0; j < result->cols; ++j)
            {
                if (!fillComboEntry(card, static_cast<PlayedStatus>(j), combo))
                    continue;
                result->set(i, static_cast<PlayedStatus>(j));
            }
        }
        return result;
    }

    std::shared_ptr<ComboTable> ComboTable::emptyTable()
    {
        std::shared_ptr<ComboTable> result = std::make_shared<ComboTable>();
        return result;
    }

    Combo ComboTable::createComboFromTableEntry(i32 loc, i32 pst)
    {
        Combo res;
        Card c;
        PlayedStatus s;
        //
        if (pst < 0 || pst >= MAX_PLAYED_STATUS) return res;
        if (!c.fromLocation(loc)) return res;
        s = static_cast<PlayedStatus>(pst);
        fillComboEntry(c, s, res);
        return res;
    }

} /* namespace regi */
