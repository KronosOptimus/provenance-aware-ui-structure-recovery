"""Model 1 type head: an MLP over per-element metadata features.

Numeric features (box, raw tree position, flags) are concatenated with a learned embedding of the
Android class name and, optionally, the mean embedding of its camel-case pieces. Output: 24 logits
(0 = invalid, 1..23 = CLAY types). Elements are classified independently (no screen context); context
arrives with the fusion model (Models 4-6).
"""
import torch
from torch import nn

from uistruct.data.features import N_CLASSES, PAD


class TypeMLP(nn.Module):
    def __init__(self, n_num, n_names, n_pieces, use_name=True, use_pieces=False, emb_dim=32,
                 hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.use_name, self.use_pieces = use_name, use_pieces
        d = n_num
        if use_name:
            self.name_emb = nn.Embedding(n_names, emb_dim)
            d += emb_dim
        if use_pieces:
            self.piece_emb = nn.Embedding(n_pieces, emb_dim, padding_idx=PAD)
            d += emb_dim
        blocks = []
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*blocks)
        self.out = nn.Linear(d, N_CLASSES)

    def forward(self, num, name, pieces):
        x = [num]
        if self.use_name:
            x.append(self.name_emb(name))
        if self.use_pieces:
            e = self.piece_emb(pieces)                                   # (B, P, D), PAD rows are 0
            n = (pieces != PAD).sum(1, keepdim=True).clamp(min=1)
            x.append(e.sum(1) / n)
        return self.out(self.body(torch.cat(x, 1)))
