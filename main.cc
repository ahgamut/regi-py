#include <regi.h>

int main()
{
    regi::GameState g;
    g.init();
    g.logDebug();
    g.startLoop();
    return 0;
}
