#include <regi.h>
#include <console.h>
#include <dfsel.h>

int main()
{
    regi::ConsoleLog c;
    regi::DamageStrategy s;
    regi::GameState g(c, s, 3);
    g.init();
    g.startLoop();
    return 0;
}
