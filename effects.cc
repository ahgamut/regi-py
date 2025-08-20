#include <regi.h>

namespace regi
{
    void GameState::playerDraws(Player &player, i32 n)
    {
        for (; n > 0 && drawPile.size() != 0 && !player.full(); n--)
        {
            player.cards.push_back(drawPile.back());
            drawPile.pop_back();
        }
    }

    i32 GameState::playerDrawsOne(Player &player)
    {
        if (drawPile.size() != 0 && !player.full())
        {
            player.cards.push_back(drawPile.front());
            log.drawOne(player);
            drawPile.erase(drawPile.begin());
            return 1;
        }
        return 0;
    }

    void GameState::refreshDiscards(i32 n)
    {
        shuffle(discardPile, 0, discardPile.size());
        i32 count = 0;
        for (; n > 0 && discardPile.size() != 0; n--)
        {
            drawPile.push_back(discardPile.back());
            discardPile.pop_back();
            count += 1;
        }
        log.replenish(count);
    }

    void GameState::refreshDraws(i32 ip, i32 n)
    {
        i32 i, j;
        std::vector<i32> full(NUM_PLAYERS);
        i32 fullct = 0;

        for (j = 0; j < NUM_PLAYERS; ++j) { full[j] = 0; }
        if (ip < 0) { ip *= -1; }
        ip %= NUM_PLAYERS;
        for (i = ip; n > 0; n--)
        {
            if (playerDrawsOne(players[i]) == 0) { full[i] = 1; }
            i += 1;
            i %= NUM_PLAYERS;
            // if all players are full, stop draw
            fullct = 0;
            for (j = 0; j < NUM_PLAYERS; ++j) { fullct += (full[j] == 1); }
            if (fullct == NUM_PLAYERS) break;
        }
    }

    /* defense */
    i32 GameState::calcBlock(Enemy &enemy)
    {
        u32 epow = getPower(enemy) & SPADES_BLOCK;
        for (auto &combo : usedPile)
        {
            if ((combo.getPowers() & JOKER_NERF) != 0) { epow = 0; }
        }
        if (epow != 0)
        {
            /* STAB: enemy is a spade, and combo does not nerf */
            return 0;
        }

        i32 blk = 0;
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
        i32 block = calcBlock(enemy);
        i32 damage = enemy.strength() - block;
        if (damage <= 0) { return; }
        i32 tblock = 0;
        for (auto c : player.cards) { tblock += c.strength(); }
        if (damage > tblock)
        {
            /* impossible to block the damage, so game over */
            log.failBlock(player, damage, tblock);
            gameOver(BLOCK_FAILED);
            player.alive = false;
            return;
        }
        selectDefense(player, damage);
    }

    /* attack */

    i32 GameState::calcDamage(Enemy &enemy)
    {
        u32 epow = getPower(enemy) & CLUBS_DOUBLE;
        for (auto &combo : usedPile)
        {
            if ((combo.getPowers() & JOKER_NERF) != 0) { epow = 0; }
        }

        i32 dmg = 0;
        Combo curcombo = usedPile.back();
        if (curcombo.parts.size() == 0)
        {
            // yielding
            pastYieldsInARow += 1;
        }
        else { pastYieldsInARow = 0; }
        dmg += curcombo.getBaseDamage();
        bool dbl = ((curcombo.getPowers() & CLUBS_DOUBLE) & (~epow)) != 0;
        if (dbl) { dmg += curcombo.getBaseDamage(); }
        return dmg;
    }

    void GameState::postAttackEffects(Player &player, Enemy &enemy)
    {
        u32 epow = getPower(enemy);
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
        u32 cpow = (curcombo.getPowers() & (~epow));
        i32 cval = (curcombo.getBaseDamage());

        if ((cpow & HEARTS_REPLENISH) != 0) { refreshDiscards(cval); }
        else if ((cpow & DIAMONDS_DRAW) != 0) { refreshDraws(player.id, cval); }
        // else do nothing
    }

    i32 GameState::enemyDead()
    {
        Enemy &enemy = enemyPile.front();
        if (enemy.hp > 0) return 0;
        Card econ(enemy.entry(), enemy.suit());
        log.enemyKill(enemy, *this);
        if (enemy.hp == 0)
        {
            /* exact kill, so add to top of draw pile */
            drawPile.insert(drawPile.begin(), econ);
            enemyPile.erase(enemyPile.begin());
        }
        else
        {
            discardPile.push_back(econ);
            enemyPile.erase(enemyPile.begin());
        }
        for (auto &comb : usedPile)
        {
            for (auto &c : comb.parts) { discardPile.push_back(c); }
        }
        usedPile.clear();
        return 1;
    }

    void GameState::attackPhase(Player &player, Enemy &enemy)
    {
        selectAttack(player, pastYieldsInARow < (NUM_PLAYERS - 1));
        i32 damage = calcDamage(enemy);
        enemy.hp -= damage;
        log.attack(player, enemy, usedPile.back(), damage);
        postAttackEffects(player, enemy);
    }

    void GameState::gameOver(EndGameReason e)
    {
        gameRunning = false;
        log.endgame(e, *this);
    }

    void GameState::postGameResult() { log.postgame(*this); }

    void GameState::startLoop()
    {
        i32 i;
        while (gameRunning)
        {
            log.state(*this);
            for (i = 0; i < NUM_PLAYERS; ++i)
            {
                oneTurn(players[i]);
                if (!gameRunning) break;
            }
            currentRound += 1;
            log.endTurn(*this);
        }
        postGameResult();
    }

    void GameState::oneTurn(Player &player)
    {
        if (!player.alive)
        {
            gameOver(PLAYER_DEAD);
            return;
        }
        if (enemyPile.size() == 0)
        {
            gameOver(NO_ENEMIES);
            return;
        }

        do {
            Enemy &enemy = enemyPile.front();
            attackPhase(player, enemy);
        } while (enemyDead());

        if (!player.alive)
        {
            gameOver(PLAYER_DEAD);
            return;
        }
        if (enemyPile.size() == 0)
        {
            gameOver(NO_ENEMIES);
            return;
        }
        defensePhase(player, enemyPile.front());
    }
} /* namespace regi */
