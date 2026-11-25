"""Provenance network training and scoring: Model 4's network + trust head on screens with wrong metadata.

  trust_arrays   per token: trust truth (1 faithful, 0 not, -1 none) and whether the token is a metadata claim
                 (elements of a screen with metadata, raw candidates). Screens of the absent level carry no claim.
  ProvenanceEncoder     Model 4's TokenEncoder (pool M, metadata on) fitted on one drawn level per training screen
                 (the distribution training sees); adds the trust arrays
  collate_fusion    collate + padded trust arrays
  train_fusion      each epoch draws one level per training screen (fixed weights, seeded); loss = candidate BCE
                 + type weight x leaf type CE + trust weight x (trust BCE after block 1 + after the last block);
                 early stopping on validation candidate AP at the real level (Model 4's rule)
  predict_fusion    candidate, leaf type and final trust probabilities per screen
  removal_tree   assembly with elements removed (plan §3-4): removed elements leave every candidate set; removed
                 valid leaves hang under the root when scored
Test is never touched here.
"""
import time

import numpy as np
import torch
from torch import nn

from uistruct.eval.metrics import VROOT
from uistruct.models.nesting import assemble
from uistruct.train.nesting_head import (CPU, TokenEncoder, batches, cand_ap, collate, loader, to_device)
from uistruct.models.fusion import FusionScorer


def trust_arrays(sc):
    """(trust_y, trust_tok) over the screen's tokens, leaves first then pool-M candidates."""
    m = sc["M"]
    if sc.get("has_meta", True):
        ly, lt = sc["leaf_trust"].astype(np.float32), np.ones(len(sc["leaf_ids"]), bool)
    else:
        ly, lt = np.full(len(sc["leaf_ids"]), -1, np.float32), np.zeros(len(sc["leaf_ids"]), bool)
    ct = m["meta"][:, 0] > 0                                          # raw candidates: the metadata's own nodes
    return np.concatenate([ly, m["trust"].astype(np.float32)]), np.concatenate([lt, ct])


class ProvenanceEncoder(TokenEncoder):
    def __init__(self, min_count=5):
        super().__init__("M", True, min_count)

    def floor_std(self):
        """A column constant in training (e.g. the masked flags when training sees only the real level) keeps
        std 1, not 1e-6: training inputs are unchanged (still 0), and a non-constant value at evaluation is not
        blown up a million times. Applied on every transform, so encoders saved before this fix are covered."""
        self.ls = np.where(self.ls < 1e-3, 1.0, self.ls)
        self.cs = np.where(self.cs < 1e-3, 1.0, self.cs)

    def fit(self, screens):
        super().fit(screens)
        self.floor_std()
        return self

    def transform(self, sc, crop=None):
        self.floor_std()
        out = super().transform(sc, crop)
        out["trust_y"], out["trust_tok"] = trust_arrays(sc)
        return out


def collate_fusion(items):
    b = collate(items)
    Lm, Cm = b["leaf_mask"].shape[1], b["cand_mask"].shape[1]
    B = len(items)
    b["trust_y"] = torch.full((B, Lm + Cm), -1.0)
    b["trust_tok"] = torch.zeros(B, Lm + Cm, dtype=torch.bool)
    for i, it in enumerate(items):
        L, C = len(it["leaf_num"]), len(it["cand_num"])
        idx = torch.from_numpy(np.r_[0:L, Lm:Lm + C])
        b["trust_y"][i, idx] = torch.from_numpy(it["trust_y"])
        b["trust_tok"][i, idx] = torch.from_numpy(it["trust_tok"])
    return b


def build_fusion(hp, enc, crop_dim, trust):
    return FusionScorer(enc.n_leaf_num, enc.n_cand_num, len(enc.names) + 2, len(enc.pieces) + 2, use_meta=True,
                        crop_dim=crop_dim, d=hp["d"], heads=hp["heads"], layers=hp["layers"], ffn=hp["ffn"],
                        dropout=hp["dropout"], trust=trust)


def draw_levels(n, levels, weights, rng):
    """One level index per screen (weights renormalised over the levels the screen set offers)."""
    w = np.array([weights[l] for l in levels], float)
    return rng.choice(len(levels), size=n, p=w / w.sum())


def predict_fusion(model, items, budget, device=CPU, workers=0):
    model.eval()
    n = len(items)
    cand, typ, trust = [None] * n, [None] * n, [None] * n
    order = batches(items, budget)
    with torch.no_grad():
        for idx, b in zip(order, loader(items, order, workers, collate_fusion)):
            out = model(to_device(b, device))
            pc, pt = torch.sigmoid(out[0]).float().cpu().numpy(), out[1].float().softmax(-1).cpu().numpy()
            pr = torch.sigmoid(out[2][1]).float().cpu().numpy() if len(out) == 3 else None
            Lm = b["leaf_mask"].shape[1]
            for r, i in enumerate(idx):
                L, C = len(items[i]["leaf_num"]), len(items[i]["cand_num"])
                cand[i], typ[i] = pc[r, :C], pt[r, :L]
                if pr is not None:
                    trust[i] = np.concatenate([pr[r, :L], pr[r, Lm:Lm + C]])
    return cand, typ, trust


def train_fusion(hp, enc, tr_levels, levels, weights, dv, seed, crop_dim, trust, trust_weight, log=print, device=CPU,
              workers=0):
    """tr_levels: per training screen, a list of transformed items in the order of `levels`.
    dv: transformed validation screens at the real level (selection)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_fusion(hp, enc, crop_dim, trust).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    best, best_state, hist, wait = -1.0, None, [], 0
    for ep in range(hp["max_epochs"]):
        t0 = time.time()
        pick = draw_levels(len(tr_levels), levels, weights, rng)
        tr = [lv[k] for lv, k in zip(tr_levels, pick)]
        model.train()
        tot, n_c, tot_t, n_t = 0.0, 0, 0.0, 0
        for b in loader(tr, batches(tr, hp["budget"], rng), workers, collate_fusion):
            b = to_device(b, device)
            out = model(b)
            lc, lt = out[0], out[1]
            m = b["cand_mask"]
            loss = bce(lc[m], b["y"][m]) / m.sum()
            if hp["type_weight"]:
                loss = loss + hp["type_weight"] * ce(lt.reshape(-1, lt.shape[-1]), b["leaf_label"].reshape(-1)) / b["leaf_mask"].sum()
            if trust:
                k = b["trust_y"] >= 0
                if k.any():
                    lt_ = sum(bce(t[k], b["trust_y"][k]) for t in out[2]) / k.sum()
                    loss = loss + trust_weight * lt_
                    tot_t += lt_.item() * int(k.sum())
                    n_t += int(k.sum())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["clip"])
            opt.step()
            tot += loss.item() * int(m.sum())
            n_c += int(m.sum())
        probs, _, _ = predict_fusion(model, dv, hp["budget"], device, workers)
        ap = cand_ap(dv, probs)
        h = {"epoch": ep + 1, "train_loss": tot / n_c, "dev_ap": ap, "sec": round(time.time() - t0, 1),
             "levels_drawn": {l: int((pick == i).sum()) for i, l in enumerate(levels)}}
        if trust:
            h["train_trust_loss"] = tot_t / max(n_t, 1)
            h["trust_bias"] = model.trust_bias.detach().cpu().numpy().round(4).tolist()
        hist.append(h)
        log(f"  ep {ep + 1:2d} loss {tot / n_c:.4f} dev AP {ap:.4f} ({time.time() - t0:.1f}s)")
        if ap > best:
            best, wait = ap, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return model, hist


# ---------------------------------------------------------------- assembly with element removal
def removal_tree(sc, scores, t_c, removed):
    """Tree over the kept elements (removed ones leave every candidate set); removed valid leaves hang under
    the root so the tree can be scored against the truth (every true leaf must appear)."""
    lids = sc["leaf_ids"].tolist()
    rem = {l for l, r in zip(lids, removed) if r}
    kept = set(lids) - rem
    cands = []
    for k, (s, x) in enumerate(zip(sc["M"]["sets"], scores)):
        s2 = s - rem
        if s2:
            cands.append((("c", k), frozenset(s2), float(x)))
    par = assemble(cands, kept, t_c)
    valid = set(sc["true"]) - set(sc["true"].values())
    for l in rem & valid:
        par[l] = VROOT
    return par
