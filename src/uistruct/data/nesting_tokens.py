"""Per-screen inputs for the candidate-container nesting scorer of Models 4-6.

One screen = a set of tokens: the given leaves, then the candidate containers of the condition's pool
(src/uistruct/models/nesting.py, pools fixed by the candidate-pool check):
  pool "P"  generated candidates only (leaf boxes alone). Pixels-only condition (Model 5).
  pool "M"  raw view-hierarchy candidates + generated ones whose leaf set no raw node has
            (Models 4 and 6). Raw candidates are every `box_ok` raw node holding a given leaf, never
            filtered by `is_labelled` (that would leak CLAY's labelling choice).

Feature groups (numeric; everything fitted later, on train only):
  leaf  box     cx, cy, w, h, log area, log aspect of the leaf box (as features.py)        every condition
  leaf  meta    raw depth, log(1 + raw child count), the nine raw flags; + class name        metadata only
  cand  geo     box around the member leaves (6), log #members, share of screen leaves,      every condition
                fill (member leaf area / union area), log(1 + non-member leaves centred inside)
  cand  src     which generators proposed the set (one flag per generator), log #generators  every condition
  cand  struct  log(1 + #crossing / #strict supersets / #strict subsets / #equal sets) in the pool
  cand  meta    is_raw, raw box (6), raw depth, log raw children, nine flags, log stack size,  metadata only
                stack position (0 = outermost of equal sets), best IoU with any raw set,
                IoU of raw box with the member-leaf box, mean member-leaf raw depth - raw depth
Model 5 never sees a raw node or its box (raw borders would leak the answer): generated candidate boxes are boxes
around their leaves. Leaf order is position (top, left), never the view-hierarchy order.

Labels (train truth = raw CLAY tree): a candidate is positive when the Hungarian one-to-one matching
of true containers to candidates on leaf-set IoU pairs it with a true container at IoU >= min_iou
(the same matching `grouping_metrics` scores). Leaves carry their CLAY type (aux type head).
"""
import numpy as np

from uistruct.data.features import FLAGS
from uistruct.eval.metrics import VROOT
from uistruct.models.nesting import generated_candidates, leaf_sets_of, oracle_scores, xy_cut
from uistruct.models.nesting_raw import prune, raw_chain

BOXF = ["cx", "cy", "w", "h", "log_area", "log_aspect"]
LEAF_META = ["depth", "log_children", *FLAGS]
CAND_GEO = [f"u_{c}" for c in BOXF] + ["log_members", "member_share", "fill", "log_intruders"]
CAND_STRUCT = ["log_cross", "log_super", "log_sub", "log_equal"]
CAND_META = (["is_raw"] + [f"r_{c}" for c in BOXF] + ["r_depth", "r_log_children"] + [f"r_{f}" for f in FLAGS]
             + ["log_stack", "stack_pos", "max_iou_raw", "iou_rawbox_union", "depth_gap"])


def src_names(pool_cfg):
    """Generator names used as source flags (axis order of XY-cut merged into one flag per tolerance)."""
    names = []
    if pool_cfg.get("singletons", True):
        names.append("single")
    if pool_cfg.get("all_leaves", True):
        names.append("all")
    names += [f"xy{t}" for t in pool_cfg.get("xy_tols", [])]
    names += [f"agglo_{m}" for m in pool_cfg.get("agglo_methods", [])]
    return names


def cand_src_cols(pool_cfg):
    return [f"src_{s}" for s in src_names(pool_cfg)] + ["log_sources"]


def box_feats(b):
    """(n, 4) normalised boxes -> (n, 6) box features, same definition as features.load_elements."""
    b = np.asarray(b, np.float64).reshape(-1, 4).clip(-0.5, 1.5)
    w, h = np.maximum(b[:, 2] - b[:, 0], 1e-4), np.maximum(b[:, 3] - b[:, 1], 1e-4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, w, h, np.log(w * h), np.log(w / h)], 1)


def box_iou(a, b):
    ix = np.maximum(0, np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]))
    iy = np.maximum(0, np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]))
    inter = ix * iy
    area = lambda x: np.maximum(x[:, 2] - x[:, 0], 0) * np.maximum(x[:, 3] - x[:, 1], 0)  # noqa: E731
    return inter / np.maximum(area(a) + area(b) - inter, 1e-12)


def set_relations(member):
    """member: (C, L) bool. Returns counts (C, C) of shared leaves and set sizes (C,)."""
    a = member.astype(np.int32)
    return a @ a.T, a.sum(1)


def struct_feats(member):
    inter, size = set_relations(member)
    si, sj = size[:, None], size[None, :]
    equal = (inter == si) & (inter == sj)
    sub = (inter == si) & (si < sj)            # i strictly inside j  -> j is a superset of i
    sup = (inter == sj) & (sj < si)
    cross = (inter > 0) & ~equal & ~sub & ~sup
    np.fill_diagonal(equal, False)
    return np.stack([np.log1p(cross.sum(1)), np.log1p(sub.sum(1)), np.log1p(sup.sum(1)),
                     np.log1p(equal.sum(1))], 1)


def _pool_arrays(sets, lids, lbox, lcent, srcs, src_list):
    """Shared candidate arrays for a list of leaf sets (geometry, sources, structure, membership)."""
    pos = {l: i for i, l in enumerate(lids)}
    L, C = len(lids), len(sets)
    member = np.zeros((C, L), bool)
    for k, s in enumerate(sets):
        member[k, [pos[l] for l in s]] = True
    big = np.where(member[:, :, None], lbox[None], np.nan)
    ub = np.stack([np.nanmin(big[..., 0], 1), np.nanmin(big[..., 1], 1),
                   np.nanmax(big[..., 2], 1), np.nanmax(big[..., 3], 1)], 1)
    uarea = np.maximum((ub[:, 2] - ub[:, 0]) * (ub[:, 3] - ub[:, 1]), 1e-8)
    larea = np.maximum(lbox[:, 2] - lbox[:, 0], 0) * np.maximum(lbox[:, 3] - lbox[:, 1], 0)
    fill = (member * larea[None]).sum(1) / uarea
    inside = ((lcent[None, :, 0] >= ub[:, None, 0]) & (lcent[None, :, 0] <= ub[:, None, 2])
              & (lcent[None, :, 1] >= ub[:, None, 1]) & (lcent[None, :, 1] <= ub[:, None, 3]))
    nm = member.sum(1)
    geo = np.column_stack([box_feats(ub), np.log(nm), nm / L, fill.clip(0, 2),
                           np.log1p((inside & ~member).sum(1))])
    src = np.zeros((C, len(src_list) + 1))
    sidx = {s: j for j, s in enumerate(src_list)}
    for k, ss in enumerate(srcs):
        names = {x.split("_")[0] if x.startswith("xy") else x for x in ss}
        for x in names:
            src[k, sidx[x]] = 1
        src[k, -1] = np.log1p(len(ss))
    return member, ub, geo, src, struct_feats(member)


def build_screen(g, raw, pool_cfg, min_iou=0.5, pools=("P", "M")):
    """g: valid elements of one screen (node_idx, gt_parent, label, nx0..ny1).
    raw: all raw nodes of the screen (node_idx, parent_idx, box_ok, nx0..ny1, depth, n_children,
    class_name, flags). Returns a dict of numpy arrays (see module docstring)."""
    true = dict(zip(g.node_idx.tolist(), g.gt_parent.tolist()))
    leaves = set(true) - set(true.values())
    # candidates are generated exactly as in the candidate-pool check (node order, unclipped boxes), so the pool is the
    # one whose ceiling was measured; tokens are then laid out in position order (top, left)
    by_node = g[g.node_idx.isin(leaves)].sort_values("node_idx")
    gen = generated_candidates(by_node.node_idx.tolist(), by_node[["nx0", "ny0", "nx1", "ny1"]].to_numpy(), pool_cfg)
    lf = by_node.assign(cy_=(by_node.ny0 + by_node.ny1) / 2, cx_=(by_node.nx0 + by_node.nx1) / 2)
    lf = lf.sort_values(["cy_", "cx_", "node_idx"])
    lids = lf.node_idx.tolist()
    lbox = lf[["nx0", "ny0", "nx1", "ny1"]].to_numpy(np.float64).clip(-0.5, 1.5)
    lcent = np.column_stack([(lbox[:, 0] + lbox[:, 2]) / 2, (lbox[:, 1] + lbox[:, 3]) / 2])
    rw = raw.set_index("node_idx")
    lmeta = np.column_stack([rw.loc[lids, "depth"].to_numpy(float),
                             np.log1p(rw.loc[lids, "n_children"].to_numpy(float)),
                             rw.loc[lids, FLAGS].to_numpy(float)])
    out = {"screen_id": int(g.screen_id.iloc[0]), "leaf_ids": np.array(lids), "leaf_box": box_feats(lbox),
           "leaf_meta": lmeta, "leaf_name": _short(rw.loc[lids, "class_name"]),
           "leaf_label": lf.label.to_numpy(np.int64), "true": true, "leaf_xyxy": lbox}
    true_sets = [s for _, s in leaf_sets_of(true, leaves)]
    src_list = src_names(pool_cfg)

    if "P" in pools:
        sets, srcs = list(gen.keys()), list(gen.values())
        member, ub, geo, src, st = _pool_arrays(sets, lids, lbox, lcent, srcs, src_list)
        y_iou = oracle_scores(true_sets, sets, min_iou)
        out["P"] = {"sets": sets, "member": member, "geo": geo, "src": src, "struct": st,
                    "y": (y_iou > 0).astype(np.float32), "y_iou": y_iou,
                    "crop_xyxy": ub}                                   # crop region: box around the leaves

    if "M" in pools:
        chain = prune(raw_chain(raw, leaves), leaves)
        raw_sets = leaf_sets_of(chain, leaves)                    # [(("raw", n), set)], chain order
        raw_vals = {s for _, s in raw_sets}
        gen_extra = [(s, v) for s, v in gen.items() if s not in raw_vals]
        cids = [c for c, _ in raw_sets] + [("gen", k) for k in range(len(gen_extra))]
        sets = [s for _, s in raw_sets] + [s for s, _ in gen_extra]
        srcs = [gen.get(s, []) for _, s in raw_sets] + [v for _, v in gen_extra]
        member, ub, geo, src, st = _pool_arrays(sets, lids, lbox, lcent, srcs, src_list)
        nr = len(raw_sets)
        rn = [c[1] for c, _ in raw_sets]
        rbox = rw.loc[rn, ["nx0", "ny0", "nx1", "ny1"]].to_numpy(np.float64).clip(-0.5, 1.5).reshape(-1, 4)
        meta = np.zeros((len(sets), len(CAND_META)))
        meta[:nr, 0] = 1
        if nr:
            meta[:nr, 1:7] = box_feats(rbox)
            meta[:nr, 7] = rw.loc[rn, "depth"].to_numpy(float)
            meta[:nr, 8] = np.log1p(rw.loc[rn, "n_children"].to_numpy(float))
            meta[:nr, 9:9 + len(FLAGS)] = rw.loc[rn, FLAGS].to_numpy(float)
            # stacks: raw candidates with the same leaf set; position 0 = outermost (smallest raw depth)
            j0 = 9 + len(FLAGS)
            groups = {}
            for k in range(nr):
                groups.setdefault(sets[k], []).append(k)
            for ks in groups.values():
                ks = sorted(ks, key=lambda k: meta[k, 7])
                for r, k in enumerate(ks):
                    meta[k, j0] = np.log(len(ks))
                    meta[k, j0 + 1] = r / max(len(ks) - 1, 1)
            meta[:nr, j0 + 3] = box_iou(rbox, ub[:nr])
            mdepth = (member[:nr] * lmeta[None, :, 0]).sum(1) / member[:nr].sum(1)
            meta[:nr, j0 + 4] = mdepth - meta[:nr, 7]
        inter, size = set_relations(member)
        if nr:
            union = size[:, None] + size[None, :nr] - inter[:, :nr]
            meta[:, 9 + len(FLAGS) + 2] = (inter[:, :nr] / np.maximum(union, 1)).max(1)
        # raw-tree edges between tokens: index of the raw parent among candidates (-1 = root / none)
        cpos = {c: k for k, c in enumerate(cids[:nr])}
        leaf_rawpar = np.array([cpos.get(chain[l], -1) for l in lids])
        cand_rawpar = np.array([cpos.get(chain[c], -1) for c in cids[:nr]] + [-1] * len(gen_extra))
        y_iou = oracle_scores(true_sets, sets, min_iou)
        out["M"] = {"sets": sets, "member": member, "geo": geo, "src": src, "struct": st, "meta": meta,
                    "name": _short(rw.loc[rn, "class_name"]) + [""] * len(gen_extra),
                    "leaf_rawpar": leaf_rawpar, "cand_rawpar": cand_rawpar,
                    "y": (y_iou > 0).astype(np.float32), "y_iou": y_iou,
                    # crop region: the raw box the metadata claims for raw candidates,
                    # the box around the leaves for generated ones. Pool P never holds a raw box.
                    "crop_xyxy": np.vstack([rbox, ub[nr:]]) if nr else ub}
    return out


def _short(class_names):
    return [("" if not isinstance(n, str) else n.split(".")[-1]) for n in class_names]


def rule_trees(g, raw, xy_tol):
    """Rule baselines on the same screen, computed exactly as the nesting baselines and the candidate-pool check: XY-cut over the leaf boxes
    (node order, unclipped) and, with metadata, the raw VH with single-child wrappers collapsed."""
    from uistruct.eval.metrics import smooth_single_child
    true = dict(zip(g.node_idx.tolist(), g.gt_parent.tolist()))
    leaves = set(true) - set(true.values())
    lf = g[g.node_idx.isin(leaves)].sort_values("node_idx")
    tops, edges = xy_cut(list(lf.node_idx), lf[["nx0", "ny0", "nx1", "ny1"]].to_numpy(), xy_tol)
    out = {"xycut": {**edges, **{t: VROOT for t in tops}}}
    if raw is not None:
        out["raw_chain_collapsed"] = smooth_single_child(prune(raw_chain(raw, leaves), leaves), leaves)
    return out
