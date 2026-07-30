"""CardTransformerNet -- one token per card, mixed by a Transformer encoder.

Roadmap Option B. Each of the 56 cards becomes a token whose features are the
window frames' ``[location | used_pile | capability]`` (``features.fuse_card_tokens``);
the tokens are embedded, given a learned per-card (card-identity) positional
embedding, mixed by a ``TransformerEncoder`` (global content-based card-to-card
attention), then read out by the shared per-card heads. Unlike ``AttnTrunkNet``
(which bolts attention onto BasicNet's conv trunk) the ENTIRE trunk is attention
over card tokens -- no convolutions on the bogus-locality card axis at all.

The encoder uses ``dropout=0.0`` so single-sample self-play ``predict`` behaves
identically to training (LayerNorm is already train/eval invariant).
"""
from regi_py.core import MAX_CARDS_IN_GAME
from regi_py.rl import features
from regi_py.rl.az.nets.base import BaseNet
from regi_py.rl.subnets import PerCardHeads

import torch
import torch.nn as nn

D_MODEL = 64
N_HEADS = 8
N_LAYERS = 3
FF_DIM = 128


class CardTransformerNet(BaseNet):
    __mname__ = "cardtx"
    max_history = 8
    TRAIN_FIELDS = ("tokens", "value", "keepyness", "atk_probs", "attacking")

    def __init__(self):
        super().__init__()
        in_dim = self.max_history * features.FEATURE_WIDTH
        self.embed = nn.Linear(in_dim, D_MODEL)
        # per-card identity positional embedding (which card each token is)
        self.card_emb = nn.Parameter(torch.randn(1, MAX_CARDS_IN_GAME, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.heads = PerCardHeads(D_MODEL)

    @classmethod
    def _assemble(cls, loc, usp, cap):
        tok = features.fuse_card_tokens(loc, usp, cap)  # (56, window*FEATURE_WIDTH)
        return {"tokens": torch.from_numpy(tok).unsqueeze(0)}

    def forward(self, data):
        x = self.embed(data["tokens"]) + self.card_emb   # (N, 56, D_MODEL)
        x = self.encoder(x)                              # (N, 56, D_MODEL)
        return self.heads(x)
