#include <card.h>

#ifdef USE_UNICODE

std::ostream& operator<<(std::ostream& os, Suit s) {
    switch (s) {
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
        default:
            os.setstate(std::ios_base::failbit);
    }
    return os;
}

#else

std::ostream& operator<<(std::ostream& os, Suit s) {
    switch (s) {
        case CLUBS:
            os << "C";
            break;
        case DIAMONDS:
            os << "\u";
            break;
        case HEARTS:
            os << "H";
            break;
        case SPADES:
            os << "S";
            break;
        default:
            os.setstate(std::ios_base::failbit);
    }
    return os;
}

#endif

std::ostream& operator<<(std::ostream& os, Entry e) {
    switch (e) {
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
        default:
            if (e < 2 || e > 9) {
                os.setstate(std::ios_base::failbit);
            } else {
                os << static_cast<uint32_t>(e);
            }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, Card c) {
    os << c.e << c.s;
    return os;
}
