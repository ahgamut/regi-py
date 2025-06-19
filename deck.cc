#include <deck.h>

#include <random>

Deck::Deck()
    : cards{
          Card(ACE, SPADES),      //
          Card(TWO, SPADES),      //
          Card(THREE, SPADES),    //
          Card(FOUR, SPADES),     //
          Card(FIVE, SPADES),     //
          Card(SIX, SPADES),      //
          Card(SEVEN, SPADES),    //
          Card(EIGHT, SPADES),    //
          Card(NINE, SPADES),     //
          Card(TEN, SPADES),      //
          Card(JACK, SPADES),     //
          Card(QUEEN, SPADES),    //
          Card(KING, SPADES),     //
                                  /* new suit */
          Card(ACE, HEARTS),      //
          Card(TWO, HEARTS),      //
          Card(THREE, HEARTS),    //
          Card(FOUR, HEARTS),     //
          Card(FIVE, HEARTS),     //
          Card(SIX, HEARTS),      //
          Card(SEVEN, HEARTS),    //
          Card(EIGHT, HEARTS),    //
          Card(NINE, HEARTS),     //
          Card(TEN, HEARTS),      //
          Card(JACK, HEARTS),     //
          Card(QUEEN, HEARTS),    //
          Card(KING, HEARTS),     //
                                  /* new suit */
          Card(ACE, DIAMONDS),    //
          Card(TWO, DIAMONDS),    //
          Card(THREE, DIAMONDS),  //
          Card(FOUR, DIAMONDS),   //
          Card(FIVE, DIAMONDS),   //
          Card(SIX, DIAMONDS),    //
          Card(SEVEN, DIAMONDS),  //
          Card(EIGHT, DIAMONDS),  //
          Card(NINE, DIAMONDS),   //
          Card(TEN, DIAMONDS),    //
          Card(JACK, DIAMONDS),   //
          Card(QUEEN, DIAMONDS),  //
          Card(KING, DIAMONDS),   //
                                  /* new suit */
          Card(ACE, CLUBS),       //
          Card(TWO, CLUBS),       //
          Card(THREE, CLUBS),     //
          Card(FOUR, CLUBS),      //
          Card(FIVE, CLUBS),      //
          Card(SIX, CLUBS),       //
          Card(SEVEN, CLUBS),     //
          Card(EIGHT, CLUBS),     //
          Card(NINE, CLUBS),      //
          Card(TEN, CLUBS),       //
          Card(JACK, CLUBS),      //
          Card(QUEEN, CLUBS),     //
          Card(KING, CLUBS)       /* end of deck */
      } {
    //
}

Deck Deck::standard() { return Deck(); }

void Deck::show() {
    for (int i = 0; i < 52; ++i) {
        std::cout << cards[i] << " ";
        if (i % 13 == 12) {
            std::cout << "\n";
        }
    }
}

void Deck::shuffle() {
    int i, j;
    std::random_device dev;
    std::default_random_engine engine(dev());
    for (i = 0; i < 50; ++i) {
        j = i + (engine() % (52 - i));
        std::swap(cards[i], cards[j]);
    }
}
