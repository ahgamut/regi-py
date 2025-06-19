#ifndef DECK_H
#define DECK_H
#include <card.h>

class Deck {
    private:
        Card cards[52];
        Deck();

    public:
        static Deck standard();
        void shuffle();
        void show();
};

#endif
