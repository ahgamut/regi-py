#include <regi.h>
#include <logger.h>
#include <dfsel.h>

int main()
{
    regi::ConsoleLog c;
    regi::RandomStrategy s;
    regi::GameState g(c, s);
    g.init();
    g.startLoop();
    return 0;
}
