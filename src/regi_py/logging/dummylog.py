from regi_py.core import BaseLog


class DummyLog(BaseLog):
    def __init__(self):
        super().__init__()

    ####
    def startgame(self, game):
        pass

    def endgame(self, reason, game):
        pass

    def postgame(self, game):
        pass

    ####

    def attack(self, player, enemy, combo, damage, game):
        pass

    def redirect(self, player, next_playerid, game):
        pass

    def defend(self, player, combo, damage, game):
        pass

    def failBlock(self, player, damage, maxblock, game):
        pass

    def fullBlock(self, player, damage, block, game):
        pass

    def drawOne(self, player):
        pass

    def cannotDrawDeckEmpty(self, player, game):
        pass

    def replenish(self, n_cards):
        pass

    def enemyKill(self, enemy, game):
        pass

    ####

    def state(self, game):
        pass

    def debug(self, game):
        pass
