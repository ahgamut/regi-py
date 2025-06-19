#include <deck.h>

int main()
{
    Card c1(Entry::ACE, Suit::CLUBS);
    std::cout << "We're here: " << c1 << "\tcard size is " << sizeof(c1) << "\n";

    Deck d1 = Deck::standard();
    d1.show();
    std::cout << "shuffling ... \n";
    d1.shuffle();
    d1.show();
    std::cout << "Deck size: " << sizeof(d1) << "\n";
    return 0;
}
