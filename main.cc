#include <regi.h>

int main()
{
    Card c1(Entry::ACE, Suit::CLUBS);
    std::cout << "We're here: " << c1 << "\tcard size is " << sizeof(c1)
              << " strength is " << c1.strength() << "\n";

    regi::GameState g;
    g.init();

    std::cout << "Player 0: " << g.players[0];
    std::cout << "Player 1: " << g.players[1];
    std::cout << "draw pile has " << g.drawPile.size() << " cards: " << g.drawPile;
    std::cout << "discard pile has " << g.discardPile.size()
              << " cards: " << g.discardPile;
    std::cout << "used pile has " << g.usedPile.size() << " combos: " << g.usedPile;
    std::cout << "enemies: " << g.enemyPile;

    return 0;
}
