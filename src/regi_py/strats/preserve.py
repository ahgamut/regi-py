from regi_py.core import BaseStrategy
import random


class PreserveStrategy(BaseStrategy):
    __strat_name__ = "preserve"

    def setup(self, player, game):
        return 0

    def get_good_attacks(self, player, combos, game):
        e = game.enemy_pile[0]
        cur_block = game.get_current_block(e)
        ac_map = {str(x): x for x in player.cards}
        all_cards = set(x for x in player.cards)

        good_indices = []
        for ind, x in enumerate(combos):
            dmg = game.get_combo_damage(e, x)
            cset = set(c for c in x.parts)
            remain = list(c for c in (all_cards - cset))
            new_blk = game.get_combo_block(e, x)
            remain_blk = sum(c.strength for c in remain)
            # print(x, "attacks for", dmg, "blocks for", new_blk)
            # print("we already block for", cur_block)
            # print("remaining cards can block at most", remain_blk)

            if remain_blk >= e.strength - (cur_block + new_blk):
                good_indices.append(ind)

        good_indices = list(
            sorted(
                good_indices,
                reverse=True,
                key=lambda i: game.get_combo_damage(e, combos[i]),
            )
        )
        # print(len(combos) - len(good_indices), "combos were removed")
        return good_indices

    def getAttackIndex(self, combos, player, yield_allowed, game):
        if len(combos) == 0:
            return -1
        good_indices = self.get_good_attacks(player, combos, game)
        if len(good_indices) > 0:
            return random.choice(good_indices) ## ???
        return random.randint(0, len(combos) - 1)

    def getDefenseIndex(self, combos, player, damage, game):
        if len(combos) == 0:
            return -1
        e = game.enemy_pile[0]
        good_indices = list(
            sorted(
                (n for n in range(len(combos))),
                key=lambda i: combos[i].base_defense
            )
        )
        return good_indices[0]
