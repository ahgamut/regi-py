#include <regi.h>
#include <logger.h>

int main()
{
    regi::ConsoleLog c;
    regi::GameState g(c);
    g.init();
    g.startLoop();
    return 0;
}
