#ifndef CARD_H
#define CARD_H
#include <cstdint>
#include <iostream>

typedef std::int32_t i32;
typedef std::uint32_t u32;
typedef std::size_t u64;

enum Suit : i32
{
    GLITCH = 0,
    CLUBS = 1,
    DIAMONDS = 2,
    HEARTS = 3,
    SPADES = 4
};
std::ostream& operator<<(std::ostream& os, const Suit s);

enum Entry : i32
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
std::ostream& operator<<(std::ostream& os, const Entry e);

struct Card
{
   private:
    Entry e;
    Suit s;

   public:
    i32 strength() const;
    Entry entry() const;
    Suit suit() const;
    Card(Entry ee, Suit ss);
    bool operator==(const Card& other) const
    {
        return (this->e == other.e) && (this->s == other.s);
    }
    friend std::ostream& operator<<(std::ostream& os, const Card& c);
};

enum Powers : u32
{
    CLUBS_DOUBLE = 1,
    DIAMONDS_DRAW = 2,
    HEARTS_REPLENISH = 4,
    SPADES_BLOCK = 8,
    JOKER_NERF = 16
};

u32 getPower(const Card& c);

#endif
