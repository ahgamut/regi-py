"""MoveTokenNet -- reason in move space, keeping the (56, 22) action contract.

Roadmap Option C, the ``derived-from-(56,22)`` variant (the full pointer form is a
larger, later step). Every legal move -- one of the 286 valid ``(location,
played_status)`` cells -- becomes a token. A move token's input feature is the
masked mean of its member cards' encoded features (a move's cards are the set bits
of its ``bitwise``, via ``features.move_structure``), plus a learned per-move
identity embedding; the move tokens are mixed by a small Transformer so the net can
compare alternative moves, then scored to one logit each. The logits scatter back
into the ``56 x 22`` grid (each move at its own cell) and go through the same masked
softmax as every other net, so the ``atk_probs`` (56, 22) target, the structural
mask and the ``(56, 22)`` predict contract are all unchanged -- and ``az.explorer``
/ ``AZNodeInfo`` need no changes.

Value and keepyness stay card-space heads over the per-card encoder features (the
same encoder that feeds the move tokens); only the ACTION head reasons over moves.
"""
from regi_py.core import MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS
from regi_py.rl import features
from regi_py.rl.az.nets.base import BaseNet

import torch
import torch.nn as nn

DIM = 64
N_MOVE_LAYERS = 2
N_HEADS = 8
FF_DIM = 128
_GRID = MAX_CARDS_IN_GAME * int(MAX_PLAYED_STATUS)


class MoveTokenNet(BaseNet):
    __mname__ = "movetoken"
    max_history = 8
    TRAIN_FIELDS = ("tokens", "value", "keepyness", "atk_probs", "attacking")

    def __init__(self):
        super().__init__()
        in_dim = self.max_history * features.FEATURE_WIDTH
        # per-card encoder: features for the card-space value/keepy heads AND the
        # source features aggregated into move tokens
        self.card_mlp = nn.Sequential(
            nn.Linear(in_dim, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
        )

        # static move structure (which cards each move uses, which grid cell it is)
        cell_flat, card_idx, card_mask = features.move_structure()
        self.n_moves = cell_flat.shape[0]
        self.register_buffer("move_cell_flat", torch.from_numpy(cell_flat), persistent=False)
        self.register_buffer("move_card_idx", torch.from_numpy(card_idx), persistent=False)
        self.register_buffer(
            "move_card_mask", torch.from_numpy(card_mask).view(1, self.n_moves, -1, 1),
            persistent=False,
        )
        # base action grid: -inf everywhere, so cells no move scatters into stay
        # masked (softmax -> 0), exactly the structurally-invalid cells
        neg_inf = torch.full((1, _GRID), float("-inf"))
        self.register_buffer("neg_inf_grid", neg_inf, persistent=False)

        # learned per-move identity embedding + move-space Transformer trunk
        self.move_emb = nn.Parameter(torch.randn(1, self.n_moves, DIM) * 0.02)
        self.move_proj = nn.Linear(DIM, DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            batch_first=True,
        )
        self.move_encoder = nn.TransformerEncoder(layer, num_layers=N_MOVE_LAYERS)
        self.move_logit = nn.Linear(DIM, 1)

        # card-space value / keepyness heads (same contract as PerCardHeads)
        self.keepy = nn.Linear(DIM, 1)
        self.value = nn.Linear(DIM, 1)

    @classmethod
    def _assemble(cls, loc, usp, cap):
        tok = features.fuse_card_tokens(loc, usp, cap)  # (56, window*FEATURE_WIDTH)
        return {"tokens": torch.from_numpy(tok).unsqueeze(0)}

    def forward(self, data):
        cardfeat = self.card_mlp(data["tokens"])   # (N, 56, DIM)
        n = cardfeat.shape[0]

        # build move tokens: masked mean of each move's member card features
        gathered = cardfeat[:, self.move_card_idx, :]        # (N, M, P, DIM)
        summed = (gathered * self.move_card_mask).sum(dim=2)  # (N, M, DIM)
        denom = self.move_card_mask.sum(dim=2).clamp_min(1.0)  # (1, M, 1)
        move_feat = summed / denom                            # (N, M, DIM)

        mt = self.move_proj(move_feat) + self.move_emb        # (N, M, DIM)
        mt = self.move_encoder(mt)                            # (N, M, DIM)
        logits = self.move_logit(mt).squeeze(-1)              # (N, M)

        # scatter move logits into the (56 x 22) grid, masked softmax over it
        grid = self.neg_inf_grid.expand(n, -1).clone()
        grid[:, self.move_cell_flat] = logits
        a = torch.softmax(grid, dim=-1).reshape(n, 1, MAX_CARDS_IN_GAME, MAX_PLAYED_STATUS)

        k = torch.sigmoid(self.keepy(cardfeat).squeeze(-1))   # (N, 56)
        v = torch.tanh(self.value(cardfeat.mean(dim=1)))      # (N, 1)
        return v, k, a
