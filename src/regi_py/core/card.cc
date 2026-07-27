#include <card.h>

static_assert(TOTAL_SUIT_OPTIONS == static_cast<i32>(SPADES) + 1);
static_assert(TOTAL_ENTRY_OPTIONS == static_cast<i32>(KING) + 1);

Card::Card(Entry ee, Suit ss) : e(ee), s(ss) {};

Entry Card::entry() const { return this->e; }

Suit Card::suit() const { return this->s; }

i32 Card::strength() const
{
    i32 st = 0;
    switch (e)
    {
        case KING:
            st = 20;
            break;
        case QUEEN:
            st = 15;
            break;
        case JACK:
            st = 10;
            break;
        default:
            st = static_cast<i32>(e);
    }
    return st;
}

i32 Card::toLocation() const
{
    /* dense bijection over all 56 cards: location = entry + 14 * suit */
    return static_cast<i32>(this->e) +
           (TOTAL_ENTRY_OPTIONS * static_cast<i32>(this->s));
}

bool Card::fromLocation(i32 loc)
{
    if (loc < 0 || loc >= MAX_CARDS_IN_GAME) return false;
    this->e = static_cast<Entry>(loc % TOTAL_ENTRY_OPTIONS);
    this->s = static_cast<Suit>(loc / TOTAL_ENTRY_OPTIONS);
    return true;
}

bool Card::isYield() const { return this->e == JOKER && this->s == CLUBS; }

bool Card::isResign() const { return this->e == JOKER && this->s == DIAMONDS; }

bool Card::operator<(const Card& other) const
{
    if (this->s == other.s) { return this->e < other.e; }
    return this->s < other.s;
}

bool Card::operator>(const Card& other) const
{
    if (this->s == other.s) { return this->e > other.e; }
    return this->s > other.s;
}

bool Card::operator==(const Card& other) const
{
    return (this->s == other.s) && (this->e == other.e);
}
#ifdef USE_UNICODE

std::ostream& operator<<(std::ostream& os, const Suit s)
{
    switch (s)
    {
        case CLUBS:
            os << "\u2663";
            break;
        case DIAMONDS:
            os << "\u2662";
            break;
        case HEARTS:
            os << "\u2661";
            break;
        case SPADES:
            os << "\u2660";
            break;
    }
    return os;
}

#else

std::ostream& operator<<(std::ostream& os, const Suit s)
{
    switch (s)
    {
        case CLUBS:
            os << "C";
            break;
        case DIAMONDS:
            os << "D";
            break;
        case HEARTS:
            os << "H";
            break;
        case SPADES:
            os << "S";
            break;
    }
    return os;
}

#endif

std::ostream& operator<<(std::ostream& os, const Entry e)
{
    switch (e)
    {
        case ACE:
            os << "A";
            break;
        case TEN:
            os << "T";
            break;
        case JACK:
            os << "J";
            break;
        case QUEEN:
            os << "Q";
            break;
        case KING:
            os << "K";
            break;
        case JOKER:
            os << "X";
            break;
        default:
            if (e < 2 || e > 9) { os.setstate(std::ios_base::failbit); }
            else { os << static_cast<u32>(e); }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Card& c)
{
    os << c.e << c.s;
    return os;
}

u32 getPower(const Card& c)
{
    /* a joker only ever nerfs the enemy; it never grants its suit's power */
    if (c.entry() == JOKER) { return JOKER_NERF; }
    u32 p = 0;
    switch (c.suit())
    {
        case CLUBS:
            p |= CLUBS_DOUBLE;
            break;
        case DIAMONDS:
            p |= DIAMONDS_DRAW;
            break;
        case HEARTS:
            p |= HEARTS_REPLENISH;
            break;
        case SPADES:
            p |= SPADES_BLOCK;
            break;
    }
    return p;
}

