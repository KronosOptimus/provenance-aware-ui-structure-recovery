"""Evaluation metrics. This file is the reference code for every metric in the report.

Type head:   macro-F1 over the 23 real types, invalid-detection F1, per-class F1, legacy 25-class
             macro-F1 (ROOT + BACKGROUND + 23, kept only for continuity with the 0.4448 baseline).
Tree head:   per-screen edge F1, leaf-edge F1, container match (Screen Parsing [3] §6.1.3:
             containers matched by IoU of their descendant sets) and edit distance.
Provenance:  ECE, reliability bins, Brier score, AUROC.

Trees are dicts {child: parent} over element ids; parent -1 is the virtual root (see data/tree.py).
Elements are given (oracle boxes), so predicted and true nodes share ids and no box matching is needed.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

TYPES = list(range(1, 24))          # 23 real types
LEGACY = list(range(-1, 24))        # ROOT, BACKGROUND, 23 types
INVALID = 0
VROOT = -1
DENSITY_BINS = [(0, 16), (16, 32), (32, 48), (48, 64), (64, None)]  # Screen Parsing's bins, [lo, hi)


# ---------------------------------------------------------------- type head
def confusion(y_true, y_pred, labels):
    """Confusion matrix over `labels`; values outside `labels` are ignored on both sides."""
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    t = np.array([idx.get(v, -1) for v in np.asarray(y_true).tolist()])
    p = np.array([idx.get(v, -1) for v in np.asarray(y_pred).tolist()])
    cm = np.zeros((k + 1, k + 1), dtype=np.int64)  # last row/col = "other"
    np.add.at(cm, (np.where(t < 0, k, t), np.where(p < 0, k, p)), 1)
    return cm


def f1_from_confusion(cm, n_labels):
    """Per-class F1 for the first n_labels classes of an (n+1)x(n+1) matrix (last = other)."""
    tp = np.diag(cm)[:n_labels].astype(float)
    fp = cm[:, :n_labels].sum(0) - tp
    fn = cm[:n_labels, :].sum(1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    return f1


def macro_f1(cm, n_labels):
    """Mean F1 over classes present in truth or prediction (absent-everywhere classes are skipped)."""
    return float(np.nanmean(f1_from_confusion(cm, n_labels)))


def type_metrics(y_true, y_pred):
    """y_true / y_pred: labels for all non-ROOT elements (0 = invalid, 1..23 = types)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion(y_true, y_pred, TYPES)
    per = f1_from_confusion(cm, len(TYPES))
    inv_t, inv_p = y_true == INVALID, y_pred == INVALID
    tp = int((inv_t & inv_p).sum())
    inv_f1 = 2 * tp / (inv_t.sum() + inv_p.sum()) if (inv_t.sum() + inv_p.sum()) else float("nan")
    valid = ~inv_t
    return {
        "macro_f1_types": float(np.nanmean(per)),
        "invalid_f1": float(inv_f1),
        "accuracy_valid": float((y_true[valid] == y_pred[valid]).mean()) if valid.any() else float("nan"),
        "per_class_f1": {int(c): (None if np.isnan(f) else round(float(f), 4)) for c, f in zip(TYPES, per)},
        "n": int(len(y_true)),
    }


def legacy_macro_f1(y_true, y_pred):
    """25-class macro-F1 including ROOT, as the proposal-era baseline (0.4448) computed it."""
    return macro_f1(confusion(y_true, y_pred, LEGACY), len(LEGACY))


# ---------------------------------------------------------------- tree head
def is_forest(parents: dict) -> bool:
    """True if following parents from every node reaches the virtual root without a cycle."""
    state = {}
    for n in parents:
        path = []
        while n != VROOT and n not in state:
            if n not in parents:
                return False  # parent is not an element
            state[n] = 0
            path.append(n)
            n = parents[n]
        if n != VROOT and state.get(n) == 0:
            return False
        for m in path:
            state[m] = 1
    return True


def _children(parents):
    ch = {}
    for c, p in parents.items():
        ch.setdefault(p, []).append(c)
    return ch


def _descendants(parents):
    """{container: frozenset(descendants)} for every node with children, virtual root excluded."""
    ch = _children(parents)
    memo = {}

    def desc(n):
        if n not in memo:
            s = set()
            for c in ch.get(n, []):
                s.add(c)
                s |= desc(c)
            memo[n] = frozenset(s)
        return memo[n]

    return {n: desc(n) for n in ch if n != VROOT}


def _f1(pred_set, true_set):
    if not pred_set and not true_set:
        return float("nan")
    tp = len(pred_set & true_set)
    return 2 * tp / (len(pred_set) + len(true_set))


def tree_metrics(true_parents: dict, pred_parents: dict) -> dict:
    """Per-screen structure metrics. Both trees must be forests under the virtual root.

    edge_f1:       F1 of (child, parent) edges.
    leaf_edge_f1:  F1 of edges whose child is a leaf (true leaves for recall, predicted for precision).
    container_match: Screen Parsing's CM. True and predicted containers (nodes with children) are paired
                   one-to-one by Hungarian matching on IoU of descendant sets; CM = sum of matched IoU /
                   number of true containers. NaN if the true tree has no container.
    edit_distance: node + edge insertions/deletions, cost 1 each, under the oracle node correspondence.
    """
    if not is_forest(pred_parents):
        raise ValueError("prediction is not a tree: assemble it before scoring")
    te, pe = set(true_parents.items()), set(pred_parents.items())
    t_leaf = set(true_parents) - set(true_parents.values())
    p_leaf = set(pred_parents) - set(pred_parents.values())
    tc, pc = _descendants(true_parents), _descendants(pred_parents)
    if tc:
        tk, pk = list(tc), list(pc)
        iou = np.zeros((len(tk), max(len(pk), 1)))
        for i, a in enumerate(tk):
            for j, b in enumerate(pk):
                iou[i, j] = len(tc[a] & pc[b]) / len(tc[a] | pc[b])
        r, c = linear_sum_assignment(-iou)
        cm = float(iou[r, c].sum() / len(tk))
    else:
        cm = float("nan")
    nodes_sym = len(set(true_parents) ^ set(pred_parents))
    return {
        "edge_f1": _f1(pe, te),
        "leaf_edge_f1": _f1({e for e in pe if e[0] in p_leaf}, {e for e in te if e[0] in t_leaf}),
        "container_match": cm,
        "edit_distance": float(len(te ^ pe) + nodes_sym),
        "n_elements": len(true_parents),
    }


def _depths(parents):
    memo = {VROOT: 0}

    def d(n):
        if n not in memo:
            memo[n] = d(parents[n]) + 1
        return memo[n]

    return {n: d(n) for n in parents}


def _leaf_sets(parents):
    """{container: frozenset(leaf descendants)} for every node with children, virtual root excluded."""
    leaves = set(parents) - set(parents.values())
    return {n: frozenset(d & leaves) for n, d in _descendants(parents).items()}


def smooth_single_child(parents: dict, keep=frozenset()) -> dict:
    """Remove containers with exactly one child (the child moves up), as Screen Parsing §5.1.3 does
    for nodes that are not visible elements. Use on the true tree when containers must be created.
    Nodes in `keep` (e.g. given leaves inside a predicted tree) are never removed."""
    p = dict(parents)
    while True:
        ch = _children(p)
        single = [n for n, c in ch.items() if n != VROOT and len(c) == 1 and n not in keep]
        if not single:
            return p
        n = single[0]
        p[ch[n][0]] = p.pop(n)


def grouping_metrics(true_parents: dict, pred_parents: dict, min_iou=0.5) -> dict:
    """Tree metrics when the prediction *creates* its own containers.

    Leaves (nodes with no children in the true tree) are given and share ids; predicted containers
    may have any ids. Every container is described by the set of leaves under it (Screen Parsing [3]
    §6.1: "Container nodes are matched ... where the score is the IoU of their descendant nodes").
    True and predicted containers are paired one-to-one by Hungarian matching on leaf-set IoU.
    container_match = sum of matched IoU / number of true containers (NaN if none).
    container_precision = sum of matched IoU / number of predicted containers (CM alone cannot see
    extra containers; the raw view hierarchy holds every true container, so over-proposing scores CM 1.0).
    container_f1 = 2 * sum of matched IoU / (true + predicted containers), the harmonic mean of the two.
    For edge F1 and edit distance, a predicted container takes the id of its matched true container
    when IoU >= min_iou (Screen Parsing's 0.5 node-matching threshold, §5.1.1); otherwise it stays
    unmatched and all its edges count as wrong. Given elements keep their ids, so when every node is
    given, edge F1 equals tree_metrics' edge F1.
    """
    if not is_forest(pred_parents):
        raise ValueError("prediction is not a tree: assemble it before scoring")
    leaves = set(true_parents) - set(true_parents.values())
    if not leaves <= set(pred_parents):
        raise ValueError("every given leaf must appear in the prediction")
    tc, pc = _leaf_sets(true_parents), _leaf_sets(pred_parents)
    tk, pk = list(tc), list(pc)
    rename, matched = {}, 0.0
    if tk and pk:
        iou = np.array([[len(tc[a] & pc[b]) / len(tc[a] | pc[b]) for b in pk] for a in tk])
        # Nested single-child containers share a leaf set, so ties are common. Break them by same id
        # (given containers), then by similar depth; the tie-break never outweighs an IoU difference.
        dt, dp = _depths(true_parents), _depths(pred_parents)
        tie = np.array([[1e-6 * (type(a) is type(b) and a == b) - 1e-9 * abs(dt[a] - dp[b]) for b in pk] for a in tk])
        r, c = linear_sum_assignment(-(iou + tie))
        matched = float(iou[r, c].sum())
        cm = matched / len(tk)
        # only created containers are renamed; a given element keeps its id, and no true id is reused
        rename = {pk[j]: tk[i] for i, j in zip(r, c)
                  if iou[i, j] >= min_iou and pk[j] not in true_parents and tk[i] not in pred_parents}
    else:
        cm = float("nan") if not tk else 0.0
    unmatched = {b: ("pred", b) for b in pk if b not in rename and b not in true_parents}
    name = lambda n: rename.get(n, unmatched.get(n, n))  # noqa: E731
    pe = {(name(c), name(p)) for c, p in pred_parents.items()}
    te = set(true_parents.items())
    nodes_sym = len(set(true_parents) ^ {name(n) for n in pred_parents})
    return {
        "edge_f1": _f1(pe, te),
        "leaf_edge_f1": _f1({e for e in pe if e[0] in leaves}, {e for e in te if e[0] in leaves}),
        "container_match": cm,
        "container_precision": matched / len(pk) if pk else float("nan"),
        "container_f1": 2 * matched / (len(tk) + len(pk)) if (tk or pk) else float("nan"),
        "edit_distance": float(len(te ^ pe) + nodes_sym),
        "n_true_containers": len(tk),
        "n_pred_containers": len(pk),
    }


def density_bin(n):
    for lo, hi in DENSITY_BINS:
        if n >= lo and (hi is None or n < hi):
            return f"{lo}-{hi}" if hi else f"{lo}+"


# ---------------------------------------------------------------- provenance / calibration
def calibration(conf, correct, n_bins=15):
    """Expected calibration error with equal-width bins, plus the bins for a reliability plot.

    conf: predicted probability that the event is true (e.g. "this node's metadata is faithful");
    correct: 1 if it was true. ECE = sum_b (n_b / N) |acc_b - conf_b|.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            acc, cf = correct[m].mean(), conf[m].mean()
            ece += m.mean() * abs(acc - cf)
            bins.append({"lo": k / n_bins, "hi": (k + 1) / n_bins, "n": int(m.sum()),
                         "conf": float(cf), "acc": float(acc)})
    auroc = roc_auc_score(correct, conf) if 0 < correct.mean() < 1 else float("nan")
    return {"ece": float(ece), "brier": float(((conf - correct) ** 2).mean()), "auroc": float(auroc),
            "bins": bins}
