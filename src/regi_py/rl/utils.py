from regi_py.core import *
import numpy as np


class Numberizer:
    MAX_SUIT_VALUE = 4.0
    MAX_ENTRY_VALUE = 15.0
    MAX_CARD_STRENGTH = 20.0
    MAX_RAW_DAMAGE = 40.0
    MAX_ENEMY_HP = 40.0
    MAX_DECK_SIZE = 52.0
    MAX_COMBO_SIZE = 4.0
    #
    MAX_HAND_SIZE = 8
    MAX_ATTACK_SIZE = 4
    ENEMY_INFO_SIZE = 5
    BLOCK_INFO_SIZE = 6
    AUXDA_INFO_SIZE = 6
    CARD_INFO_SIZE = 4
    OTHERP_INFO_SIZE = 3
    MAX_ENEMIES = 12
    MAX_PLAYERS = 4

    def __init__(self):
        pass

    def numberize_card(self, card):
        res = np.zeros(self.CARD_INFO_SIZE, dtype=np.float32)
        res[0] = 1.0  # indicates valid card
        res[1] = card.suit.value / self.MAX_SUIT_VALUE
        res[2] = card.entry.value / self.MAX_ENTRY_VALUE
        res[3] = card.strength / self.MAX_CARD_STRENGTH
        return res

    def numberize_enemy(self, enemy):
        res = np.zeros(self.ENEMY_INFO_SIZE, dtype=np.float32)
        res[0] = enemy.hp > 0  # indicates alive
        res[1] = enemy.suit.value / self.MAX_SUIT_VALUE
        res[2] = enemy.entry.value / self.MAX_ENTRY_VALUE
        res[3] = enemy.strength / self.MAX_CARD_STRENGTH
        res[4] = enemy.hp / self.MAX_ENEMY_HP
        return res

    def numberize_attack(self, combo):
        # combo can at most be 4 cards(2, 2, 2, 2)
        res = np.zeros((self.MAX_ATTACK_SIZE, self.CARD_INFO_SIZE), dtype=np.float32)
        for i, part in enumerate(combo.parts):
            res[i, :] = self.numberize_card(part)
        return res

    def numberize_other_player(self, player):
        res = np.zeros(self.OTHERP_INFO_SIZE, dtype=np.float32)
        res[0] = 1 if player.alive else -1
        res[1] = 0  # how far is this player from current state ?
        res[2] = len(player.cards) / self.MAX_HAND_SIZE
        return res

    def numberize_other_players(self, player, game):
        res = np.zeros((self.MAX_PLAYERS, self.OTHERP_INFO_SIZE), dtype=np.float32)
        j = 0
        for op in game.players:
            res[j, :] = self.numberize_other_player(op)
            d = op.id - player.id
            while d < 0:
                d += game.num_players
            d = d % game.num_players
            res[j, 1] = d
        return res

    def numberize_current_enemy(self, game):
        res = np.zeros(self.BLOCK_INFO_SIZE, dtype=np.float32)
        e = None
        if len(game.enemy_pile) > 0:
            e = game.enemy_pile[0]
            res[0] = game.get_current_block(e)
            res[1:] = self.numberize_enemy(e)
        return e, res

    def numberize_remaining_enemies(self, game):
        res = np.zeros(self.MAX_ENEMIES, dtype=np.float32)
        for i, e in enumerate(game.enemy_pile):
            res[i] = max(e.hp, 0)
        return np.sum(res)

    def numberize_used_pile(self, game):
        res = np.zeros((1 + len(game.used_combos), 16))
        for i, c in enumerate(game.used_combos):
            res[i, :] = self.numberize_attack(c).ravel()
        return res

    def numberize_aux_data(self, player, game, attacking=False):
        # auxiliary game info that may be useful?
        res = np.zeros(self.AUXDA_INFO_SIZE, dtype=np.float32)
        res[0] = attacking
        res[1] = len(game.enemy_pile) / self.MAX_ENEMIES
        res[2] = len(game.draw_pile) / self.MAX_DECK_SIZE
        res[3] = len(game.discard_pile) / self.MAX_DECK_SIZE
        res[4] = len(game.used_combos) / self.MAX_DECK_SIZE
        res[5] = game.past_yields / game.num_players
        return res

    def one_hot_combo_index(self, combo, player):
        res = np.zeros(self.MAX_HAND_SIZE, dtype=np.float32)
        for i, p1 in enumerate(combo.parts):
            for j, p2 in enumerate(player.cards):
                if p1 == p2:
                    res[j] = 1
                    break
        return res

    def numberize_attack_combos(self, combos, player, game, enemy):
        f_cdmgs = np.zeros(len(combos), dtype=np.float32)
        f_combo_inds = np.zeros((len(combos), self.MAX_HAND_SIZE), dtype=np.float32)
        for i, c in enumerate(combos):
            f_combo_inds[i, :] = self.one_hot_combo_index(c, player)
            if enemy is not None:
                f_cdmgs[i] = game.get_combo_damage(enemy, c) / self.MAX_ENEMY_HP
        return f_combo_inds, f_cdmgs

    def numberize_defend_combos(self, combos, player, game, enemy):
        f_cblks = np.zeros(len(combos), dtype=np.float32)
        f_combo_inds = np.zeros((len(combos), self.MAX_HAND_SIZE), dtype=np.float32)
        for i, c in enumerate(combos):
            f_combo_inds[i, :] = self.one_hot_combo_index(c, player)
            if enemy is not None:
                f_cblks[i] = c.base_defense / self.MAX_ENEMY_HP
        return f_combo_inds, f_cblks

    def numberize_hand(self, player):
        res = np.zeros((self.MAX_HAND_SIZE, self.CARD_INFO_SIZE), dtype=np.float32)
        for i, c in enumerate(player.cards):
            res[i, :] = self.numberize_card(c)
        return res

    def numberize_state(self, combos, player, game, attacking=False):
        #
        f_curphd = self.numberize_hand(player)
        #
        f_otherp = self.numberize_other_players(player, game)
        #
        enemy, f_enemy = self.numberize_current_enemy(game)
        #
        if attacking:
            f_indices, f_values = self.numberize_attack_combos(
                combos, player, game, enemy
            )
        else:
            f_indices, f_values = self.numberize_defend_combos(
                combos, player, game, enemy
            )
        #
        f_usedp = self.numberize_used_pile(game)
        #
        f_auxda = self.numberize_aux_data(player, game, attacking)
        #
        f_remy = self.numberize_remaining_enemies(game)
        #
        proxy = 0
        if len(game.enemy_pile) > 0:
            e0 = game.enemy_pile[0]
            if attacking:
                proxy = enemy.hp
        #
        state = {
            "status": str(game.status.name),
            "phase": game.phase_count,
            "player": player.id,
            "attacking": attacking,
            "curphd": f_curphd,
            "otherp": f_otherp,
            "enemy": f_enemy,
            "indices": f_indices,
            "values": f_values,
            "usedp": f_usedp,
            "auxda": f_auxda,
            "remaining": f_remy,
            "option": None,
            "reward": 0,
            "best_future": 0,
            "best_from_here": 0,
            "proxy": proxy
        }
        return state


class MemoryLog(BaseLog):
    def __init__(self, N=100):
        super().__init__()
        self.N = N
        self.numberizer = Numberizer()
        self.memories = []

    def record(self, cls):
        old_getAttackIndex = cls.getAttackIndex
        old_getDefenseIndex = cls.getDefenseIndex

        def rec_getAttackIndex(obj, combos, player, yield_allowed, game):
            state = self.numberizer.numberize_state(combos, player, game, True)
            option = old_getAttackIndex(obj, combos, player, yield_allowed, game)
            state["option"] = option
            if len(self.memories) >= self.N:
                self.memories.pop(0)
            self.memories.append(state)
            return option

        def rec_getDefenseIndex(obj, combos, player, damage, game):
            state = self.numberizer.numberize_state(combos, player, game, False)
            option = old_getDefenseIndex(obj, combos, player, damage, game)
            state["option"] = option
            if len(self.memories) >= self.N:
                self.memories.pop(0)
            self.memories.append(state)
            return option

        cls.getAttackIndex = rec_getAttackIndex
        cls.getDefenseIndex = rec_getDefenseIndex
        return cls

    ####
    def startgame(self, game):
        pass

    def endgame(self, reason, game):
        pass

    def postgame(self, game):
        active_player = max(0, game.active_player)
        if len(game.enemy_pile) == 0:
            print("game ends with a WIN!")
        else:
            remaining = sum(e.hp for e in game.enemy_pile)
            print("game ends at ", game.enemy_pile[0], end=" ")
            print("remaining: ", remaining, end = "\n")
        end_state = self.numberizer.numberize_state([], game.players[active_player], game, True)
        end_state["option"] = None
        self.memories.append(end_state)

    ####

    def attack(self, player, enemy, combo, damage, game):
        pass

    def defend(self, player, combo, damage, game):
        pass

    def failBlock(self, player, damage, maxblock, game):
        pass

    def drawOne(self, player):
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

    def startPlayerTurn(self, game):
        pass

    def endPlayerTurn(self, game):
        pass
