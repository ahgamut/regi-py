#ifndef CARD_H
#define CARD_H
#include <cstdint>
#include <iostream>

enum Suit : std::uint8_t { CLUBS = 1, DIAMONDS = 2, HEARTS = 3, SPADES = 4 };
std::ostream& operator<<(std::ostream& os, Suit s);

enum Entry : std::uint8_t {
    ACE = 1,
    TWO = 2,
    THREE = 3,
    FOUR = 4,
    FIVE = 5,
    SIX = 6,
    SEVEN = 7,
    EIGHT = 8,
    NINE = 9,
    TEN = 10,
    JACK = 11,
    QUEEN = 12,
    KING = 13
};
std::ostream& operator<<(std::ostream& os, Entry e);

struct Card {
   private:
    Entry e : 5;
    Suit s : 3;

   public:
    Card(Entry ee, Suit ss) : e(ee), s(ss) {};
    friend std::ostream& operator<<(std::ostream& os, Card c);
};

#endif
