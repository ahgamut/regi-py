#include <card.h>

Card::Card(Entry ee, Suit ss) : e(ee), s(ss) {};

Entry Card::entry() const { return this->e; }

Suit Card::suit() const { return this->s; }

std::int32_t Card::strength() const
{
    std::int32_t st = 0;
    switch (s)
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
            st = static_cast<std::int32_t>(s);
    }
    return st;
}

#ifdef USE_UNICODE

std::ostream& operator<<(std::ostream& os, Suit s)
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
        case GLITCH:
            os << "!";
            break;
    }
    return os;
}

#else

std::ostream& operator<<(std::ostream& os, Suit s)
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
        case GLITCH:
            os << "!";
            break;
    }
    return os;
}

#endif

std::ostream& operator<<(std::ostream& os, Entry e)
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
            else { os << static_cast<uint32_t>(e); }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, Card c)
{
    os << c.e << c.s;
    return os;
}

std::uint32_t getPower(Card& c)
{
    std::uint32_t p;
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
        case GLITCH:
            p |= JOKER_NERF;
            break;
    }
    return p;
}

