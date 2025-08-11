#ifndef CARD_H
#define CARD_H
#include <cstdint>
#include <iostream>

enum Suit : std::uint8_t
{
    GLITCH = 0,
    CLUBS = 1,
    DIAMONDS = 2,
    HEARTS = 3,
    SPADES = 4
};
std::ostream& operator<<(std::ostream& os, Suit s);

enum Entry : std::uint8_t
{
    JOKER = 0,
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

struct Card
{
   private:
    Entry e : 5;
    Suit s : 3;

   public:
    std::int32_t strength() const;
    Entry entry() const;
    Suit suit() const;
    Card(Entry ee, Suit ss);
    friend std::ostream& operator<<(std::ostream& os, Card c);
};

enum Powers : std::uint32_t
{
    CLUBS_DOUBLE = 1,
    DIAMONDS_DRAW = 2,
    HEARTS_REPLENISH = 4,
    SPADES_BLOCK = 8,
    JOKER_NERF = 16
};

std::uint32_t getPower(Card& c);

#endif
