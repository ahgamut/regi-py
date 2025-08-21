#include <regi.h>
#include <console.h>
#include <dfsel.h>

int main()
{
    regi::ConsoleLog c;
    regi::DamageStrategy s;

    std::vector<regi::Player> players;
    players.push_back(regi::Player(s));
    players.push_back(regi::Player(s));
    players.push_back(regi::Player(s));
    regi::GameState g(c, players);
    g.init();
    g.startLoop();
    return 0;
}
