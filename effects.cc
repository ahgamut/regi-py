#include <regi.h>

namespace regi
{
    void GameState::playerDraws(Player &player, int n)
    {
        for (; n > 0 && drawPile.size() != 0 && !player.full(); n--)
        {
            player.cards.push_back(drawPile.back());
            drawPile.pop_back();
        }
    }

    int GameState::playerDrawsOne(Player &player)
    {
        if (drawPile.size() != 0 && !player.full())
        {
            player.cards.push_back(drawPile.back());
            drawPile.pop_back();
            return 1;
        }
        return 0;
    }

    void GameState::refreshDiscards(int n)
    {
        shuffle(discardPile, 0, discardPile.size());
        for (; n > 0 && discardPile.size() != 0; n--)
        {
            drawPile.push_back(discardPile.back());
            discardPile.pop_back();
        }
    }

    void GameState::refreshDraws(int ip, int n)
    {
        int i;
        int full[] = {0, 0};
        if (ip < 0) { ip *= -1; }
        ip %= 2;
        for (i = ip; n > 0; n--)
        {
            if (playerDrawsOne(players[i]) == 0) { full[i] = 1; }
            i += 1;
            i %= 2;
            if (full[0] == 1 && full[1] == 1) break;
        }
    }

    /* defense */
    int GameState::calcBlock(Enemy &enemy)
    {
        std::uint32_t epow = getPower(enemy) & SPADES_BLOCK;
        for (auto &combo : usedPile)
        {
            if ((combo.getPowers() & JOKER_NERF) != 0) { epow = 0; }
        }
        if (epow != 0)
        {
            /* STAB: enemy is a spade, and combo does not nerf */
            return 0;
        }

        int blk = 0;
        for (auto &combo : usedPile)
        {
            if ((combo.getPowers() & SPADES_BLOCK) != 0)
            {
                blk += combo.getBaseDamage();
            }
        }
        return blk;
    }

    void GameState::defensePhase(Player &player, Enemy &enemy)
    {
        int block = calcBlock(enemy);
        int damage = enemy.strength() - block;
        if (damage <= 0) { return; }
        int tblock = 0;
        for (auto c : player.cards) { tblock += c.strength(); }
        if (damage > tblock)
        {
            /* impossible to block the damage, so game over */
            gameOver();
            player.alive = false;
            return;
        }
        selectDefense(player, damage);
    }

    /* attack */

    int GameState::calcDamage(Enemy &enemy)
    {
        std::uint32_t epow = getPower(enemy) & CLUBS_DOUBLE;
        for (auto &combo : usedPile)
        {
            if ((combo.getPowers() & JOKER_NERF) != 0) { epow = 0; }
        }

        int dmg = 0;
        for (auto &combo : usedPile)
        {
            dmg += combo.getBaseDamage();
            bool dbl = ((combo.getPowers() & CLUBS_DOUBLE) ^ epow) != 0;
            if (dbl) { dmg += combo.getBaseDamage(); }
        }
        return dmg;
    }

    void GameState::postAttackEffects(Player &player, Enemy &enemy)
    {
        std::uint32_t epow = getPower(enemy);
        if (enemy.hp <= 0)
        {
            /* we check effects before removing dead enemies,
             * so ignore STAB if enemy was killed just now */
            epow = 0;
        }
        for (auto &combo : usedPile)
        {
            if ((combo.getPowers() & JOKER_NERF) != 0) { epow = 0; }
        }

        Combo curcombo = usedPile.back();
        std::uint32_t cpow = (curcombo.getPowers() ^ epow);
        std::int32_t cval = (curcombo.getBaseDamage());

        if ((cpow & HEARTS_REPLENISH) != 0) { refreshDiscards(cval); }
        else if ((cpow & DIAMONDS_DRAW) != 0) { refreshDraws(player.id, cval); }
        // else do nothing
    }

    int GameState::enemyDead()
    {
        Enemy &enemy = enemyPile.front();
        if (enemy.hp > 0) return 0;
        Card econ(enemy.entry(), enemy.suit());
        if (enemy.hp == 0)
        {
            /* exact kill, so add to top of draw pile */
            drawPile.insert(drawPile.begin(), econ);
        }
        else { discardPile.push_back(econ); }
        for (auto &comb : usedPile)
        {
            for (auto &c : comb.parts) { discardPile.push_back(c); }
        }
        usedPile.clear();
        return 1;
    }

    void GameState::attackPhase(Player &player, Enemy &enemy)
    {
        selectAttack(player);
        int damage = calcDamage(enemy);
        enemy.hp -= damage;
        postAttackEffects(player, enemy);
    }

    void GameState::gameOver() { gameRunning = false; }
    void GameState::postGameResult()
    {
        if (!players[0].alive) { std::cout << "LOST! player 0 KO\n"; }
        else if (!players[1].alive) { std::cout << "LOST! player 1 KO\n"; }
        else if (!enemyPile.size() == 0) { std::cout << "WIN!\n"; }
        else { std::cout << "unknown exit\n"; }
    }

    void GameState::startLoop()
    {
        while (gameRunning)
        {
            oneTurn(players[0]);
            oneTurn(players[1]);
        }
        postGameResult();
    }

    void GameState::oneTurn(Player &player)
    {
        if (enemyPile.size() == 0 || !player.alive)
        {
            gameOver();
            return;
        }

        do {
            Enemy &enemy = enemyPile.front();
            attackPhase(player, enemy);
        } while (enemyDead());

        if (enemyPile.size() == 0 || !player.alive)
        {
            gameOver();
            return;
        }
        defensePhase(player, enemyPile.front());
    }
} /* namespace regi */
