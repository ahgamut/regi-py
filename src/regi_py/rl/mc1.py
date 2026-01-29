from regi_py.core import *
from regi_py.rl.utils import *
from regi_py.rl.subnets import LinearBlock, Conv1dBlock, Conv2dBlock
from regi_py.strats import DamageStrategy, PreserveStrategy
import random
import time

#
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

__all__ = ("MC1Model",)


def card_embedding(dim, pad_id):
    return nn.Embedding(70, embedding_dim=dim, padding_idx=pad_id)


class DrawDiscardModule(nn.Module):
    def __init__(self, target=1):
        super().__init__()
        self.net1 = nn.Conv1d(55, target, 16)
        self.znet = nn.Bilinear(target, target, 128)
        self.ac = nn.LeakyReLU(0.05)

    def forward(self, draw_pile, discard_pile):
        N = draw_pile.shape[0]
        x0 = self.net1(draw_pile).reshape(N, -1)
        x1 = self.net1(discard_pile).reshape(N, -1)
        x = self.znet(x0, x1)
        x = self.ac(x)
        x = x.reshape(N, 1, -1)
        return x


class UsedPileModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net1 = Conv2dBlock(
            channels=[16, 18, 20, 22, 24, 32],
            shapes=[(1, 5), (1, 5), (1, 5), 3, 3],
            paddings=[0, 0, 0, 1, 0],
        )

    def forward(self, pile):
        N = pile.shape[0]
        x = self.net1(pile)
        x = x.reshape(N, 1, -1)
        return x


class EnemyPileModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net1 = Conv1dBlock(
            channels=[12, 16, 20, 24, 32],  #
            shapes=[7, 5, 5, 3],  #
            paddings=[0, 0, 0, 1],  #
        )
        self.net2 = nn.Conv1d(12, dim, 1)
        self.znet = nn.Bilinear(64, dim, 128)
        self.ac = nn.Sigmoid()

    def forward(self, cards, hp):
        N = cards.shape[0]
        x0 = self.net1(cards).reshape(N, -1)
        x1 = self.net2(hp).reshape(N, -1)
        x = self.znet(x0, x1)
        x = x.reshape(N, 1, -1)
        return self.ac(x)


class PlayerCardsModule(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net1 = Conv2dBlock(
            channels=[4, 8, 12, 16, 24, 32],  #
            shapes=[(3, 7), (3, 7), 5, 3, 3],  #
            paddings=[(1, 1), (1, 1), 0, 0, 1],  #
        )
        self.net2 = nn.Embedding(2, embedding_dim=dim, padding_idx=None)
        self.znet = nn.Bilinear(128, dim, 128)
        self.ac = nn.Sigmoid()

    def forward(self, cards, atk_def):
        N = cards.shape[0]
        x0 = self.net1(cards).reshape(N, -1)
        x1 = self.net2(atk_def)
        x = self.znet(x0, x1)
        x = x.reshape(N, 1, -1)
        return self.ac(x)


class ProbModule(nn.Module):
    def __init__(self, preset=3):
        super().__init__()
        depth = preset + 2
        self.net1 = Conv1dBlock(
            channels=[4] * preset + [3, 2, 1], shapes=[3] * depth, paddings=[1] * depth
        )
        self.net2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.ac = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        x = self.net1(x)
        x = self.net2(x)
        x = self.ac(x) + 1e-8
        x = x.reshape(N, -1)
        return x


class ValueModule(nn.Module):
    def __init__(self, preset=1):
        super().__init__()
        depth = preset + 2
        self.net1 = Conv1dBlock(
            channels=[4] * preset + [3, 2, 1], shapes=[3] * depth, paddings=[1] * depth
        )
        self.net2 = nn.Linear(128, 1)
        self.ac = nn.Sigmoid()

    def forward(self, x):
        N = x.shape[0]
        x = self.net1(x)
        x = x.reshape(N, -1)
        x = self.net2(x)
        x = x.reshape(-1)
        return self.ac(x)


class MC1Model(torch.nn.Module):
    NOT_A_CARD = 1  # (GLITCH ACE)
    UNKNOWN_CARD = 2  # (GLITCH TWO)
    CARD_DIMENSION = 16

    def __init__(self):
        super(MC1Model, self).__init__()
        self.device = "cpu"
        self.gen_emb = card_embedding(self.CARD_DIMENSION, self.NOT_A_CARD)

        self.pc_mod = PlayerCardsModule(4)
        self.dd_mod = DrawDiscardModule(2)
        self.em_mod = EnemyPileModule(2)
        self.up_mod = UsedPileModule(16)
        self.prob_net = ProbModule(3)
        self.val_net = ValueModule(1)

    def forward(self, states):
        player_cards = self.gen_emb(states["player_cards"])
        N = player_cards.shape[0]
        player_atk = states["player_atk"]
        x0 = self.pc_mod(player_cards, player_atk)
        # print(player_cards.shape, x0.shape)
        #
        draw_pile = self.gen_emb(states["draw_pile"])
        discard_pile = self.gen_emb(states["discard_pile"])
        x1 = self.dd_mod(draw_pile, discard_pile)
        # print(draw_pile.shape, x1.shape)
        #
        enemy_pile = self.gen_emb(states["enemy_pile"])
        enemy_hp = states["enemy_hp"]
        x2 = self.em_mod(enemy_pile, enemy_hp)
        # print(enemy_pile.shape, x2.shape)
        #
        used_pile = self.gen_emb(states["used_pile"])
        x3 = self.up_mod(used_pile)
        # print(used_pile.shape, x3.shape)

        x4 = torch.cat([x0, x1, x2, x3], dim=1)
        prob_hat = self.prob_net(x4)
        v_hat = self.val_net(x4)
        # print(x4.shape, prob_hat.shape, v_hat.shape)

        return prob_hat, v_hat

    def predict(self, sample):
        states, _, _ = self.tensorify([sample], 1)
        prob_hat0, v_hat0 = self.forward(states)
        prob_hat = prob_hat0.cpu().detach().numpy()[0]
        v_hat = v_hat0.cpu().detach().numpy()[0]
        return prob_hat, v_hat

    def tensorify(self, examples, batch_size):
        return MC1Model._tensorify(examples, batch_size)

    @classmethod
    def _tensorify(cls, examples, batch_size):
        sub = random.sample(examples, batch_size)

        player_cards = torch.full((batch_size, 4, 8), cls.NOT_A_CARD, dtype=torch.long)
        enemy_pile = torch.full((batch_size, 12), cls.NOT_A_CARD, dtype=torch.long)
        enemy_hp = torch.zeros((batch_size, 12, 1))
        used_pile = torch.full((batch_size, 16, 4), cls.NOT_A_CARD, dtype=torch.long)
        draw_pile = torch.full((batch_size, 55), cls.NOT_A_CARD, dtype=torch.long)
        discard_pile = torch.full((batch_size, 55), cls.NOT_A_CARD, dtype=torch.long)
        player_atk = torch.zeros((batch_size,), dtype=torch.long)
        probs = torch.zeros((batch_size, 128))
        values = torch.zeros((batch_size,))

        for b, s in enumerate(sub):
            # access
            phase = s.phase
            opc = phase.player_cards
            N = phase.num_players
            cur = phase.active_player
            combi = phase.used_combos
            num_draw = len(phase.draw_pile)
            num_discard = len(phase.discard_pile)
            assert len(phase.enemy_pile) <= 12, "too many enemies"

            # active player cards
            for j, c in enumerate(opc[cur]):
                player_cards[b, 0, j] = c.index
            # other player cards
            for i in range(1, N):
                ii = (cur + i) % N
                for j, c in enumerate(opc[ii]):
                    player_cards[b, i, j] = cls.UNKNOWN_CARD
            # enemies
            for i, c in enumerate(phase.enemy_pile):
                if i == 0:
                    enemy_pile[b, i] = c.index
                else:
                    enemy_pile[b, i] = int(c.entry)  # GLITCH, ENTRY

                enemy_hp[b, i, 0] = c.hp / 40.0
            # used combos
            for i in range(min(len(combi), 16)):
                for j, c in enumerate(combi[i].parts):
                    used_pile[b, (16 - i - 1), j] = c.index
            # attacking?
            player_atk[b] = 1 if phase.phase_attacking else 0
            # draw
            if num_draw > 0:
                draw_pile[b, :num_draw] = cls.UNKNOWN_CARD
            # discard_pile
            if num_discard > 0:
                discard_pile[b, :num_discard] = cls.UNKNOWN_CARD
            # results
            if len(s.policy) > 0:
                pb = s.policy
                if len(s.combos) > 0:
                    pb *= s.policy * s.combos
                probs[b] = torch.from_numpy(pb)
            values[b] = s.value

        res = dict(
            player_cards=player_cards,
            enemy_pile=enemy_pile,
            enemy_hp=enemy_hp,
            used_pile=used_pile,
            draw_pile=draw_pile,
            discard_pile=discard_pile,
            player_atk=player_atk,
        )
        return res, probs, values
