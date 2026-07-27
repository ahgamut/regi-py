#ifndef CARD_H
#define CARD_H
#include <cstdint>
#include <iostream>

typedef std::uint8_t u8;
typedef std::uint16_t u16;
typedef std::int32_t i32;
typedef std::uint32_t u32;
typedef std::size_t u64;

constexpr i32 TOTAL_SUIT_OPTIONS = 4;
constexpr i32 TOTAL_ENTRY_OPTIONS = 14;
/* Every (entry, suit) pair is a card, located at (entry + 14 * suit), 0..55.
 * The four JOKER-entry cards are the "special" slots (location % 14 == 0):
 *   (JOKER, CLUBS)    -> location 0  -> yield
 *   (JOKER, DIAMONDS) -> location 14 -> resign  (never dealt into the game)
 *   (JOKER, HEARTS)   -> location 28 -> joker 1 (real deck joker)
 *   (JOKER, SPADES)   -> location 42 -> joker 2 (real deck joker) */
constexpr i32 MAX_CARDS_IN_GAME = 56;
constexpr i32 LOCATION_YIELD = 0;
constexpr i32 LOCATION_RESIGN = 14;
constexpr i32 LOCATION_JOKER_1 = 28;
constexpr i32 LOCATION_JOKER_2 = 42;

enum Suit : u16
{
    CLUBS = 0,
    DIAMONDS = 1,
    HEARTS = 2,
    SPADES = 3
};
std::ostream& operator<<(std::ostream& os, const Suit s);

enum Entry : u16
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
    Card() : e(KING), s(SPADES) {};
    Card(Entry ee, Suit ss);
    i32 strength() const;
    Entry entry() const;
    Suit suit() const;
    i32 toLocation() const;
    bool fromLocation(i32);
    /* the yield / resign "cards" are sentinels for those actions; they encode
     * to locations 0 / 14 and must never appear in a player's hand. */
    bool isYield() const;
    bool isResign() const;
    bool operator<(const Card&) const;
    bool operator>(const Card&) const;
    bool operator==(const Card&) const;
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
