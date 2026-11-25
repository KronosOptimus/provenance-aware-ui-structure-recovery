"""Training and scoring for the candidate-container nesting head. Settings come from configs/train_structure_models.yaml.

  TokenEncoder  fitted on training screens only: numeric means/stds, class-name and name-piece vocabularies
  collate       screens -> padded tensors + relation matrices (models/fusion.py)
  train_scorer  AdamW, BCE on candidates (+ optional type loss on leaves); early stopping on validation
                average precision of the candidate scores (threshold-free); best epoch kept
  tree_rows     assemble each screen at a threshold and score it with grouping_metrics (raw + smoothed truth)
  tune_threshold  the assembler threshold that maximises mean validation container F1 (raw truth)
  GBDT comparator: sklearn HistGradientBoosting on the same candidates, tabular features only
Test is never touched here.
"""
import copy
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from torch import nn

from uistruct.data.features import MAX_PIECES, PAD, UNK, camel_pieces
from uistruct.eval.metrics import density_bin, grouping_metrics, smooth_single_child
from uistruct.models.fusion import FusionScorer, raw_relation_matrix, relation_matrix
from uistruct.models.nesting import assemble

NO_NAME = PAD      # generated candidates have no class name: zero vector


# ---------------------------------------------------------------- inputs
def leaf_numeric(sc, use_meta):
    return np.hstack([sc["leaf_box"], sc["leaf_meta"]]) if use_meta else sc["leaf_box"]


def cand_numeric(sc, pool, use_meta):
    p = sc[pool]
    parts = [p["geo"], p["src"], p["struct"]] + ([p["meta"]] if use_meta else [])
    return np.hstack(parts)


class TokenEncoder:
    def __init__(self, pool, use_meta, min_count=5):
        self.pool, self.use_meta, self.min_count = pool, use_meta, min_count

    def fit(self, screens):
        ln = np.vstack([leaf_numeric(s, self.use_meta) for s in screens])
        cn = np.vstack([cand_numeric(s, self.pool, self.use_meta) for s in screens])
        self.lm, self.ls = ln.mean(0), ln.std(0) + 1e-6
        self.cm, self.cs = cn.mean(0), cn.std(0) + 1e-6
        self.names, self.pieces = {}, {}
        if self.use_meta:
            cnt = {}
            for s in screens:
                for n in list(s["leaf_name"]) + list(s[self.pool]["name"]):
                    if n:
                        cnt[n] = cnt.get(n, 0) + 1
            self.names = {n: i + 2 for i, n in enumerate(sorted((n for n, c in cnt.items() if c >= self.min_count),
                                                                key=lambda n: -cnt[n]))}
            pc = {}
            for n, c in cnt.items():
                for p in camel_pieces(n):
                    pc[p] = pc.get(p, 0) + c
            self.pieces = {p: i + 2 for i, p in enumerate(sorted((p for p, c in pc.items() if c >= self.min_count),
                                                                 key=lambda p: -pc[p]))}
        return self

    @property
    def n_leaf_num(self):
        return len(self.lm)

    @property
    def n_cand_num(self):
        return len(self.cm)

    def _ids(self, names):
        ids = np.array([NO_NAME if not n else self.names.get(n, UNK) for n in names], np.int64)
        pcs = np.zeros((len(names), MAX_PIECES), np.int64)
        for i, n in enumerate(names):
            ps = [self.pieces.get(p, UNK) for p in camel_pieces(n)] if n else []
            pcs[i, :len(ps)] = ps
        return ids, pcs

    def fit_crops(self, leaf_crops, cand_crops):
        """Per-dimension mean/std of the crop embeddings, training screens only (one statistic for
        leaves and candidates together, so both live in the same space)."""
        x = np.vstack([np.vstack(leaf_crops), np.vstack(cand_crops)]).astype(np.float32)
        self.crm, self.crs = x.mean(0), x.std(0) + 1e-3
        return self

    def _crop(self, c):
        return ((np.asarray(c, np.float32) - self.crm) / self.crs).astype(np.float16)

    def transform(self, sc, crop=None):
        """One screen -> dict of numpy arrays ready for collate. crop = (leaf, candidate) embeddings."""
        p = sc[self.pool]
        out = {"leaf_num": ((leaf_numeric(sc, self.use_meta) - self.lm) / self.ls).astype(np.float32),
               "cand_num": ((cand_numeric(sc, self.pool, self.use_meta) - self.cm) / self.cs).astype(np.float32),
               "member": p["member"], "y": p["y"].astype(np.float32),
               "leaf_label": sc["leaf_label"], "rel": relation_matrix(p["member"])}
        if self.use_meta:
            out["leaf_name"], out["leaf_pieces"] = self._ids(sc["leaf_name"])
            out["cand_name"], out["cand_pieces"] = self._ids(p["name"])
            out["raw_rel"] = raw_relation_matrix(p["leaf_rawpar"], p["cand_rawpar"])
        if crop is not None:                          # crop embeddings per leaf / candidate
            out["leaf_crop"], out["cand_crop"] = self._crop(crop[0]), self._crop(crop[1])
        return out


def collate(items):
    """Pad a list of transformed screens. Leaves occupy tokens [0, Lmax), candidates [Lmax, Lmax + Cmax)."""
    B = len(items)
    Lm = max(len(it["leaf_num"]) for it in items)
    Cm = max(len(it["cand_num"]) for it in items)
    N = Lm + Cm
    b = {"leaf_mask": torch.zeros(B, Lm, dtype=torch.bool), "cand_mask": torch.zeros(B, Cm, dtype=torch.bool),
         "member": torch.zeros(B, Cm, Lm, dtype=torch.bool), "y": torch.zeros(B, Cm),
         "leaf_label": torch.full((B, Lm), -100, dtype=torch.long), "rel": torch.zeros(B, N, N, dtype=torch.int8)}
    has = lambda k: k in items[0]  # noqa: E731
    for k, last in [("leaf_num", None), ("cand_num", None), ("leaf_crop", None), ("cand_crop", None)]:
        if has(k):
            n = Lm if k.startswith("leaf") else Cm
            b[k] = torch.zeros(B, n, items[0][k].shape[1])
    for k in ["leaf_name", "cand_name"]:
        if has(k):
            b[k] = torch.zeros(B, Lm if k.startswith("leaf") else Cm, dtype=torch.long)
    for k in ["leaf_pieces", "cand_pieces"]:
        if has(k):
            b[k] = torch.zeros(B, Lm if k.startswith("leaf") else Cm, MAX_PIECES, dtype=torch.long)
    if has("raw_rel"):
        b["raw_rel"] = torch.zeros(B, N, N, dtype=torch.int8)
    for i, it in enumerate(items):
        L, C = len(it["leaf_num"]), len(it["cand_num"])
        b["leaf_mask"][i, :L], b["cand_mask"][i, :C] = True, True
        b["member"][i, :C, :L] = torch.from_numpy(it["member"])
        b["y"][i, :C] = torch.from_numpy(it["y"])
        b["leaf_label"][i, :L] = torch.from_numpy(it["leaf_label"])
        idx = np.r_[0:L, Lm:Lm + C]
        ix = torch.from_numpy(idx)
        b["rel"][i][ix[:, None], ix[None, :]] = torch.from_numpy(it["rel"])
        if has("raw_rel"):
            b["raw_rel"][i][ix[:, None], ix[None, :]] = torch.from_numpy(it["raw_rel"])
        for k in ["leaf_num", "leaf_crop", "leaf_name", "leaf_pieces"]:
            if has(k):
                b[k][i, :L] = torch.from_numpy(it[k])
        for k in ["cand_num", "cand_crop", "cand_name", "cand_pieces"]:
            if has(k):
                b[k][i, :C] = torch.from_numpy(it[k])
    return b


def batches(items, budget, rng=None):
    """Index batches of similar-size screens with sum over the batch of N^2 <= budget (N = tokens)."""
    n = np.array([len(it["leaf_num"]) + len(it["cand_num"]) for it in items], float)
    key = n + (rng.uniform(0, 1, len(n)) * 4 if rng is not None else 0)     # small jitter between epochs
    order = np.argsort(key, kind="stable")
    out, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, n[i])
        if cur and m * m * (len(cur) + 1) > budget:
            out.append(cur)
            cur, m = [], n[i]
        cur.append(i)
        cur_max = m
    if cur:
        out.append(cur)
    if rng is not None:
        rng.shuffle(out)
    return out


# ---------------------------------------------------------------- transformer scorer
def build_model(hp, enc, crop_dim=0):
    return FusionScorer(enc.n_leaf_num, enc.n_cand_num, len(enc.names) + 2, len(enc.pieces) + 2,
                        use_meta=enc.use_meta, crop_dim=crop_dim, d=hp["d"], heads=hp["heads"],
                        layers=hp["layers"], ffn=hp["ffn"], dropout=hp["dropout"])


CPU = torch.device("cpu")


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def loader(items, idx_batches, workers, collate_fn=collate):
    """Collated batches in the given order; collation runs in `workers` processes (0 = in this process).
    Workers are forked on Linux (Colab), so the item list is shared, not copied."""
    if not workers:
        return (collate_fn([items[i] for i in idx]) for idx in idx_batches)
    return torch.utils.data.DataLoader(items, batch_sampler=idx_batches, collate_fn=collate_fn, num_workers=workers,
                                       prefetch_factor=4)


def predict(model, items, budget, device=CPU, workers=0):
    """Candidate probabilities per screen (list of arrays) and leaf type probabilities per screen."""
    model.eval()
    cand, typ = [None] * len(items), [None] * len(items)
    order = batches(items, budget)
    with torch.no_grad():
        for idx, b in zip(order, loader(items, order, workers)):
            lc, lt = model(to_device(b, device))
            pc, pt = torch.sigmoid(lc).float().cpu().numpy(), lt.float().softmax(-1).cpu().numpy()
            for r, i in enumerate(idx):
                cand[i] = pc[r, :len(items[i]["cand_num"])]
                typ[i] = pt[r, :len(items[i]["leaf_num"])]
    return cand, typ


def cand_ap(items, probs):
    return float(average_precision_score(np.concatenate([it["y"] for it in items]), np.concatenate(probs)))


def train_scorer(hp, enc, tr, dv, seed, crop_dim=0, log=print, device=CPU, workers=0):
    """tr / dv: lists of transformed screens. Returns (model, history). device/workers: Colab GPU."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(hp, enc, crop_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    best, best_state, hist, wait = -1.0, None, [], 0
    for ep in range(hp["max_epochs"]):
        t0 = time.time()
        model.train()
        tot, n_c = 0.0, 0
        order = batches(tr, hp["budget"], rng)
        for b in loader(tr, order, workers):
            b = to_device(b, device)
            lc, lt = model(b)
            m = b["cand_mask"]
            loss = bce(lc[m], b["y"][m]) / m.sum()
            if hp["type_weight"]:
                loss = loss + hp["type_weight"] * ce(lt.reshape(-1, lt.shape[-1]), b["leaf_label"].reshape(-1)) / b["leaf_mask"].sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["clip"])
            opt.step()
            tot += loss.item() * int(m.sum())
            n_c += int(m.sum())
        probs, _ = predict(model, dv, hp["budget"], device, workers)
        ap = cand_ap(dv, probs)
        hist.append({"epoch": ep + 1, "train_loss": tot / n_c, "dev_ap": ap, "sec": round(time.time() - t0, 1)})
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


# ---------------------------------------------------------------- GBDT comparator (same candidates)
def gbdt_features(sc, pool, use_meta, enc, crop=None):
    """Tabular rows for the trees comparator: the scorer's candidate features, screen context
    (log #leaves, log #candidates) and, with metadata, the class name as a category and the mean of the
    member leaves' raw features (trees see no other token, so they get this summary instead).
    With crops: the first `pca_dim` principal components of the candidate's own crop embedding
    and of the mean of its member leaves' crop embeddings (PCA fitted on training candidates only)."""
    p = sc[pool]
    x = [cand_numeric(sc, pool, use_meta),
         np.tile([np.log(len(sc["leaf_ids"])), np.log(len(p["sets"]))], (len(p["sets"]), 1))]
    if crop is not None:
        m = p["member"].astype(np.float32)
        lc, cc = enc.pca.transform(enc._crop(crop[0]).astype(np.float32)), enc.pca.transform(enc._crop(crop[1]).astype(np.float32))
        x += [cc, m @ lc / m.sum(1, keepdims=True)]
    if use_meta:
        m = p["member"].astype(float)
        x.append(m @ sc["leaf_meta"] / m.sum(1, keepdims=True))
        ids = np.array([enc.gbdt_cats.get(n, len(enc.gbdt_cats)) if n else len(enc.gbdt_cats) + 1 for n in p["name"]])
        x.append(ids[:, None].astype(float))
    return np.hstack(x)


def fit_gbdt(hp, screens, pool, use_meta, enc, seed, crops=None, pca_dim=32):
    if use_meta:   # most frequent raw-candidate class names on train (sklearn caps categories at 255 values)
        cnt = {}
        for s in screens:
            for n in s[pool]["name"]:
                if n:
                    cnt[n] = cnt.get(n, 0) + 1
        enc.gbdt_cats = {n: i for i, n in enumerate(sorted(cnt, key=lambda n: -cnt[n])[:hp["max_cats"]])}
    if crops is not None:          # PCA on (at most 300k) standardised training candidate crops
        from sklearn.decomposition import PCA
        cc = np.vstack([c[1] for c in crops])
        rows = np.random.default_rng(seed).permutation(len(cc))[:300_000]
        enc.pca = PCA(pca_dim, random_state=seed).fit(enc._crop(cc[np.sort(rows)]).astype(np.float32))
    X = np.vstack([gbdt_features(s, pool, use_meta, enc, None if crops is None else c)
                   for s, c in zip(screens, crops if crops is not None else [None] * len(screens))])
    y = np.concatenate([s[pool]["y"] for s in screens])
    cat = [False] * X.shape[1]
    if use_meta:
        cat[-1] = True
    clf = HistGradientBoostingClassifier(
        learning_rate=hp["learning_rate"], max_leaf_nodes=hp["max_leaf_nodes"], l2_regularization=hp["l2"],
        min_samples_leaf=hp["min_samples_leaf"], max_iter=hp["max_iter"], categorical_features=cat,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=seed)
    return clf.fit(X, y)


def gbdt_predict(clf, screens, pool, use_meta, enc, crops=None):
    cs = crops if crops is not None else [None] * len(screens)
    return [clf.predict_proba(gbdt_features(s, pool, use_meta, enc, c))[:, 1] for s, c in zip(screens, cs)]


# ---------------------------------------------------------------- assembly and tree scoring
def screen_tree(sc, pool, scores, threshold):
    p = sc[pool]
    cands = [(("c", k), s, float(x)) for k, (s, x) in enumerate(zip(p["sets"], scores))]
    return assemble(cands, set(sc["leaf_ids"].tolist()), threshold)


def tree_rows(sc, pred, system, truths=("raw", "smoothed")):
    t = sc["true"]
    base = {"screen_id": sc["screen_id"], "system": system, "n_leaves": len(sc["leaf_ids"]),
            "bin": density_bin(len(sc["leaf_ids"]))}
    rows = []
    for tname in truths:
        tt = t if tname == "raw" else smooth_single_child(t)
        rows.append({**base, "truth": tname, **grouping_metrics(tt, pred)})
    return rows


def _f1_at(args):
    sc, pool, scores, thresholds = args
    return [grouping_metrics(sc["true"], screen_tree(sc, pool, scores, t))["container_f1"] for t in thresholds]


def tune_threshold(screens, pool, probs, thresholds, mapper=map):
    """Mean validation container F1 (raw truth) per threshold; best threshold (ties -> closest to 0.5)."""
    f1 = np.array(list(mapper(_f1_at, [(s, pool, p, thresholds) for s, p in zip(screens, probs)])))
    mean = np.nanmean(f1, 0)
    best = max(range(len(thresholds)), key=lambda i: (round(mean[i], 6), -abs(thresholds[i] - 0.5)))
    return float(thresholds[best]), {f"{t:.2f}": float(m) for t, m in zip(thresholds, mean)}
