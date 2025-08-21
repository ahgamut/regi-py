#ifdef SOLO_BUILD
#include <regi.h>
#include <console.h>
#include <dfsel.h>

int main()
{
    regi::ConsoleLog c;
    regi::DamageStrategy s;
    regi::GameState g(c);

    g.addPlayer(s);
    g.addPlayer(s);
    g.addPlayer(s);
    g.init();
    g.startLoop();
    return 0;
}
#endif
