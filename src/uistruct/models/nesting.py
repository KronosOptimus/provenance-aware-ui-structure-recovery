"""Nesting by candidate containers.

The true leaf elements are given. Containers are chosen from a pool of candidates, each described
only by the set of given leaves under it (the same description `grouping_metrics` scores):
  - generated candidates, made from the leaf boxes alone (allowed in every condition):
    singletons, the all-leaves set, recursive XY-cut at several tolerances and both first axes,
    and every merge of agglomerative clustering on box-gap distance;
  - raw candidates (metadata conditions only): every raw view-hierarchy node with a usable box
    that holds at least one given leaf. Never filtered by `is_labelled` (would leak CLAY's choice).
A scorer gives every candidate a score; `assemble` keeps the high scorers and builds a valid tree.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.optimize import linear_sum_assignment

VROOT = -1
Y_SCALE = 16 / 9   # nx/ny are fractions of width/height; screens are 9:16, so this makes units square


def xy_cut(ids, b, tol, depth=0, counter=None):
    """Recursive XY-cut over leaf boxes (same rule as scripts/measure_structure_inputs.py).
    Even depth cuts along y first. Returns (top-level nodes, {child: parent} edges); new
    containers get ids ('xy', k)."""
    counter = counter if counter is not None else [0]
    if len(ids) == 1:
        return list(ids), {}
    for axis in ([1, 0] if depth % 2 == 0 else [0, 1]):
        lo, hi = b[:, axis], b[:, axis + 2]
        order = np.argsort(lo, kind="stable")
        groups, cur, end = [], [order[0]], hi[order[0]]
        for i in order[1:]:
            if lo[i] >= end - tol:
                groups.append(cur)
                cur, end = [i], hi[i]
            else:
                cur.append(i)
                end = max(end, hi[i])
        groups.append(cur)
        if len(groups) > 1:
            break
    else:
        return list(ids), {}
    tops, edges = [], {}
    for gidx in groups:
        gids, gb = [ids[i] for i in gidx], b[gidx]
        sub_tops, sub_edges = xy_cut(gids, gb, tol, depth + 1, counter)
        edges.update(sub_edges)
        if len(sub_tops) == 1:
            tops.append(sub_tops[0])
        else:
            counter[0] += 1
            cid = ("xy", counter[0])
            for t in sub_tops:
                edges[t] = cid
            tops.append(cid)
    return tops, edges


def leaf_sets_of(parents: dict, leaves) -> list:
    """Leaf set of every container (node with children, virtual root excluded), one entry per node."""
    leaves = set(leaves)
    ch = {}
    for c, p in parents.items():
        ch.setdefault(p, []).append(c)
    memo = {}

    def under(n):
        if n not in memo:
            s = {n} if n in leaves else set()
            for c in ch.get(n, []):
                s |= under(c)
            memo[n] = frozenset(s)
        return memo[n]

    return [(n, under(n)) for n in ch if n != VROOT and under(n)]


def box_gap_distance(b):
    """Condensed pairwise distance: length of the gap between two boxes (0 if they touch/overlap)."""
    b = b.astype(float).copy()
    b[:, [1, 3]] *= Y_SCALE
    gx = np.maximum(0, np.maximum(b[:, None, 0], b[None, :, 0]) - np.minimum(b[:, None, 2], b[None, :, 2]))
    gy = np.maximum(0, np.maximum(b[:, None, 1], b[None, :, 1]) - np.minimum(b[:, None, 3], b[None, :, 3]))
    d = np.sqrt(gx ** 2 + gy ** 2)
    iu = np.triu_indices(len(b), 1)
    return d[iu]


def agglomerative_sets(ids, b, method):
    """Leaf set of every merge in agglomerative clustering on box-gap distance."""
    n = len(ids)
    if n < 2:
        return []
    z = linkage(box_gap_distance(b), method=method)
    members = [frozenset([i]) for i in ids]
    for a, c, _, _ in z:
        members.append(members[int(a)] | members[int(c)])
    return members[n:]


def generated_candidates(ids, b, cfg):
    """Distinct leaf sets proposed from leaf boxes alone, with the source of each.
    Returns {frozenset: [source, ...]} so sources can be ablated."""
    out = {}

    def add(s, src):
        if s:
            out.setdefault(frozenset(s), []).append(src)

    ids = list(ids)
    if cfg.get("singletons", True):
        for i in ids:
            add({i}, "single")
    if cfg.get("all_leaves", True):
        add(ids, "all")
    for tol in cfg.get("xy_tols", []):
        for first in (0, 1):        # depth 0 = y first, depth 1 = x first
            tops, edges = xy_cut(ids, b, tol, depth=first)
            par = {**edges, **{t: VROOT for t in tops}}
            for _, s in leaf_sets_of(par, ids):
                add(s, f"xy{tol}_{'yx'[first]}")
    for m in cfg.get("agglo_methods", []):
        for s in agglomerative_sets(ids, b, m):
            add(s, f"agglo_{m}")
    return out


def assemble(cands, leaves, threshold):
    """Build a valid tree from scored candidates.

    cands: list of (cid, leaf frozenset, score). Candidates with score >= threshold are taken in
    score order; one that crosses an already kept set (overlaps it without nesting) is dropped.
    Equal leaf sets may all be kept (a stack of single-child wrappers). Each leaf hangs under the
    smallest kept set containing it; each kept set under the smallest kept set strictly around it
    (inside a stack: the higher score is outside). Returns {node: parent} with VROOT at the top.
    """
    kept = []
    for cid, s, sc in sorted((c for c in cands if c[2] >= threshold), key=lambda c: -c[2]):
        if all(s <= k or k <= s or not (s & k) for _, k, _ in kept):
            kept.append((cid, s, sc))
    order = sorted(kept, key=lambda c: (-len(c[1]), -c[2]))   # outer before inner
    par = {}
    for i, (cid, s, _) in enumerate(order):
        par[cid] = VROOT
        for j in range(i - 1, -1, -1):
            if s <= order[j][1]:
                par[cid] = order[j][0]
                break
    for leaf in leaves:
        par[leaf] = VROOT
        for j in range(len(order) - 1, -1, -1):
            if leaf in order[j][1]:
                par[leaf] = order[j][0]
                break
    return par


def oracle_scores(true_sets, cand_sets, min_iou=0.5):
    """Best possible picking: Hungarian one-to-one match of true containers to candidates on leaf-set
    IoU; a matched candidate with IoU >= min_iou scores its IoU, every other candidate 0."""
    scores = np.zeros(len(cand_sets))
    if not true_sets or not cand_sets:
        return scores
    iou = np.array([[len(t & c) / len(t | c) for c in cand_sets] for t in true_sets])
    r, c = linear_sum_assignment(-iou)
    for i, j in zip(r, c):
        if iou[i, j] >= min_iou:
            scores[j] = iou[i, j]
    return scores


def best_iou_per_true(true_sets, cand_sets):
    """For coverage: best IoU any candidate reaches for each true container (no one-to-one)."""
    if not cand_sets:
        return np.zeros(len(true_sets))
    return np.array([max(len(t & c) / len(t | c) for c in cand_sets) for t in true_sets])
