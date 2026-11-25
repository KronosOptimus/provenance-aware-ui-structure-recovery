"""Shared fusion scorer for Models 4-6 (one token set + a small transformer).

Tokens of one screen = the given leaves + the candidate containers of the condition's pool
(src/uistruct/data/nesting_tokens.py). Inputs can be switched off, so the same code serves:
  Model 4  use_meta=True,  crop_dim>0   (pool M: raw + generated candidates)
  Model 5  use_meta=False, crop_dim>0   (pool P: generated candidates only)
  Model 6  use_meta=True,  crop_dim=0
With use_meta=False the model gets no metadata feature, no class name and no raw-tree edge.

Self-attention is relation-aware: every pair of tokens gets a learned per-head bias for its set
relation (leaf in candidate, candidate inside / around / equal to / crossing / apart from another
candidate, ...). Without it a transformer could not tell which leaves a candidate holds. With
metadata, raw view-hierarchy parent/child links add a second bias. Candidates also start from the
mean of their member leaves' input vectors. No positional encoding: position is in the box features,
and token order never carries view-hierarchy order.

Outputs: one logit per candidate ("is this a true container?", scored jointly over the screen) and
24 type logits per leaf (auxiliary head; scored on leaves only).

Provenance network (trust=True): a trust head (one small MLP) reads every token after block 1 and
after the last block; its logit is p(the metadata box of this element / raw candidate is faithful). From
block 2 on, attention to a token marked in b["trust_tok"] gets a learned per-head bias times log p(trust)
from block 1 (bias starts at 0, so the network starts as Model 4). forward then returns a third output,
(trust logits after block 1, trust logits after the last block), each (B, N). With trust=False the model
and its parameter names are exactly the Models 4-6 ones, so their checkpoints still load.
"""
import math

import numpy as np
import torch
from torch import nn

from uistruct.data.features import N_CLASSES, PAD

# set relations, query token i -> key token j
REL_SELF, REL_LL, REL_L_IN, REL_L_OUT, REL_C_HAS, REL_C_NOT = 0, 1, 2, 3, 4, 5
REL_SUB, REL_SUPER, REL_EQUAL, REL_CROSS, REL_APART = 6, 7, 8, 9, 10
N_REL = 11
N_RAW_REL = 3     # 0 none, 1 = key is the query's raw parent, 2 = key is a raw child of the query


def relation_matrix(member):
    """member: (C, L) bool. Returns (L + C, L + C) int8 relations; leaves first, then candidates."""
    C, L = member.shape
    R = np.empty((L + C, L + C), np.int8)
    R[:L, :L] = REL_LL
    R[:L, L:] = np.where(member.T, REL_L_IN, REL_L_OUT)
    R[L:, :L] = np.where(member, REL_C_HAS, REL_C_NOT)
    a = member.astype(np.int32)
    inter, size = a @ a.T, a.sum(1)
    si, sj = size[:, None], size[None, :]
    cc = np.full((C, C), REL_CROSS, np.int8)
    cc[inter == 0] = REL_APART
    cc[(inter == si) & (si < sj)] = REL_SUB
    cc[(inter == sj) & (sj < si)] = REL_SUPER
    cc[(inter == si) & (inter == sj)] = REL_EQUAL
    R[L:, L:] = cc
    np.fill_diagonal(R, REL_SELF)
    return R


def raw_relation_matrix(leaf_rawpar, cand_rawpar):
    """Raw-tree links among tokens (index of the raw parent among candidates, -1 = none)."""
    L, C = len(leaf_rawpar), len(cand_rawpar)
    R = np.zeros((L + C, L + C), np.int8)
    par = np.concatenate([leaf_rawpar, cand_rawpar])
    child = np.nonzero(par >= 0)[0]
    R[child, L + par[child]] = 1
    R[L + par[child], child] = 2
    return R


class RelAttention(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, key_mask):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = q @ k.transpose(-1, -2) / math.sqrt(self.dk) + bias          # (B, H, N, N)
        att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        att = self.drop(att.softmax(-1))
        return self.out((att @ v).transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, heads, ffn, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = RelAttention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, key_mask):
        x = x + self.drop(self.att(self.n1(x), bias, key_mask))
        return x + self.drop(self.ff(self.n2(x)))


class FusionScorer(nn.Module):
    def __init__(self, n_leaf_num, n_cand_num, n_names, n_pieces, use_meta, crop_dim=0, d=128, heads=4,
                 layers=3, ffn=256, dropout=0.1, emb_dim=32, trust=False):
        super().__init__()
        self.use_meta, self.crop_dim, self.heads, self.trust = use_meta, crop_dim, heads, trust
        if trust:
            assert layers >= 2, "the trust bias acts from block 2 on"
            self.trust_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.trust_bias = nn.Parameter(torch.zeros(layers - 1, heads))
        self.leaf_in = nn.Linear(n_leaf_num, d)
        self.cand_in = nn.Linear(n_cand_num, d)
        self.member_in = nn.Linear(d, d)
        self.kind = nn.Embedding(2, d)                    # 0 leaf, 1 candidate
        if use_meta:
            self.name_emb = nn.Embedding(n_names, emb_dim, padding_idx=PAD)
            self.piece_emb = nn.Embedding(n_pieces, emb_dim, padding_idx=PAD)
            self.name_proj = nn.Linear(2 * emb_dim, d)
            self.raw_bias = nn.Embedding(N_RAW_REL, heads * layers)
        if crop_dim:
            self.crop_proj = nn.Linear(crop_dim, d)
        self.rel_bias = nn.Embedding(N_REL, heads * layers)
        self.blocks = nn.ModuleList([Block(d, heads, ffn, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.cand_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.type_head = nn.Linear(d, N_CLASSES)

    def _names(self, name, pieces):
        e = self.piece_emb(pieces)
        n = (pieces != PAD).sum(-1, keepdim=True).clamp(min=1)
        return self.name_proj(torch.cat([self.name_emb(name), e.sum(-2) / n], -1))

    def forward(self, b):
        """b: batch dict from train.nesting_head.collate. Returns (cand logits (B, C), type logits (B, L, 24))."""
        hl = self.leaf_in(b["leaf_num"]) + self.kind.weight[0]
        hc = self.cand_in(b["cand_num"]) + self.kind.weight[1]
        if self.use_meta:
            hl = hl + self._names(b["leaf_name"], b["leaf_pieces"])
            hc = hc + self._names(b["cand_name"], b["cand_pieces"])
        if self.crop_dim:
            hl = hl + self.crop_proj(b["leaf_crop"])
            hc = hc + self.crop_proj(b["cand_crop"])
        m = b["member"].float()                                               # (B, C, L)
        hc = hc + self.member_in(m @ hl / m.sum(-1, keepdim=True).clamp(min=1))
        x = torch.cat([hl, hc], 1)
        L = hl.shape[1]
        bias = self.rel_bias(b["rel"].long())                                 # (B, N, N, H * layers)
        if self.use_meta:
            bias = bias + self.raw_bias(b["raw_rel"].long())
        bias = bias.permute(0, 3, 1, 2)
        mask = torch.cat([b["leaf_mask"], b["cand_mask"]], 1)
        t1 = logp = None
        for i, blk in enumerate(self.blocks):
            bi = bias[:, i * self.heads:(i + 1) * self.heads]
            if logp is not None:                                              # keys: (B, 1, 1, N)
                bi = bi + self.trust_bias[i - 1][None, :, None, None] * logp[:, None, None, :]
            x = blk(x, bi, mask)
            if self.trust and i == 0:
                t1 = self.trust_head(x).squeeze(-1)
                logp = nn.functional.logsigmoid(t1) * b["trust_tok"].float()
        xn = self.norm(x)
        out = (self.cand_head(xn[:, L:]).squeeze(-1), self.type_head(xn[:, :L]))
        if self.trust:
            return (*out, (t1, self.trust_head(x).squeeze(-1)))
        return out
