"""Per-screen inputs for the provenance network: metadata as delivered (wrong boxes included), synthetic corruption, trust labels.

A screen is stored once per corruption level; each level is a dict shaped like a Models 4-6 screen
(data/nesting_tokens.py, pool "M" only), plus:
  leaf_trust  (L,) 1 = the element's box is faithful, 0 = not (CLAY invalid or injected fake)
  leaf_kind   (L,) 0 valid leaf, 1 CLAY invalid, 2 injected fake (analysis only, never an input)
  leaf_meta   gains a last column `masked` (1 = this element's metadata is hidden)
  M["trust"]  (C,) raw candidate truth: 1 CLAY valid type, 0 CLAY invalid, -1 none (ROOT, unlabelled, generated)
Elements = true valid leaves + CLAY invalid nodes that hold no valid leaf and no other such node in the raw
tree (deepest ones; an invalid node holding a valid leaf is already a raw candidate) + injected fakes.
Candidate labels use the candidate's set with invalid members removed (the structure scorer is not asked to
judge boxes; the trust head is). Truth tree = the ground-truth tree over valid elements, as for Models 4-6.
Labels and `is_labelled` are read only to build truth, never as inputs.
"""
import numpy as np

from uistruct.data.features import FLAGS
from uistruct.data.nesting_tokens import _pool_arrays, _short, box_feats, box_iou, set_relations, src_names
from uistruct.eval.metrics import VROOT
from uistruct.models.nesting import generated_candidates, leaf_sets_of, oracle_scores
from uistruct.models.nesting_raw import prune, raw_chain

LEVELS_TRAIN = ("real", "heavy", "heavy_noinject", "absent")
LEVEL_ID = {"real": 0, "heavy": 1, "absent": 2, "inject2": 3, "drop50": 4, "heavy_noinject": 5}   # fixed: seeds never shift
INJECT_BASE = 10 ** 6           # node ids of injected fakes start here (never a RICO pre-order index)
BOX = ["nx0", "ny0", "nx1", "ny1"]


# ---------------------------------------------------------------- which invalid nodes become elements
def _ancestors(par, nodes):
    out = set()
    for n in nodes:
        p = par.get(n, VROOT)
        while p != VROOT and p not in out:
            out.add(p)
            p = par.get(p, VROOT)
    return out


def invalid_elements(raw, valid_leaves):
    """CLAY invalid nodes with a usable box that hold no valid leaf and no other such node (raw tree)."""
    par = dict(zip(raw.node_idx, raw.parent_idx))
    inv = set(raw.node_idx[(raw.label == 0) & raw.is_labelled & raw.box_ok])
    cand = inv - _ancestors(par, valid_leaves)
    return sorted(cand - _ancestors(par, cand))


# ---------------------------------------------------------------- corruption
def donor_table(raw_all, invalid_elems=None):
    """Real invalid nodes of one split (box, class name, flags, child count) for injection.
    invalid_elems: {screen_id: [node_idx]} of the invalid *elements* (invalid_elements); fakes are copies of
    these, so they look like the real errors they sit beside (fingerprint check, run1 failed with all
    invalid nodes as donors: large invalid wrappers have other child counts and flags)."""
    d = raw_all[(raw_all.label == 0) & raw_all.is_labelled & raw_all.box_ok]
    if invalid_elems is not None:
        keep = set((s, n) for s, ns in invalid_elems.items() for n in ns)
        d = d[[k in keep for k in zip(d.screen_id, d.node_idx)]]
    return {"screen_id": d.screen_id.to_numpy(), "box": d[BOX].to_numpy(np.float64),
            "class_name": d.class_name.to_numpy(object), "flags": d[FLAGS].to_numpy(),
            "n_children": d.n_children.to_numpy()}


def inject(raw, n, donors, rng, elements=frozenset()):
    """Add n fake nodes copied from real invalid nodes of other screens. Each goes under the smallest
    usable raw node whose box holds its centre (the root if none), depth = host depth + 1. Elements are
    never hosts: an element is a leaf of the model's input tree (see elements_as_leaves)."""
    if n == 0:
        return raw, []
    sid = int(raw.screen_id.iloc[0])
    pool = np.nonzero(donors["screen_id"] != sid)[0]
    pick = pool[rng.integers(0, len(pool), n)]
    ok = raw[raw.box_ok & ~raw.node_idx.isin(elements)]
    ob = ok[BOX].to_numpy(np.float64)
    oarea = (ob[:, 2] - ob[:, 0]) * (ob[:, 3] - ob[:, 1])
    rows = []
    for k, j in enumerate(pick):
        b = donors["box"][j]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        hold = (ob[:, 0] <= cx) & (ob[:, 2] >= cx) & (ob[:, 1] <= cy) & (ob[:, 3] >= cy)
        if hold.any():
            h = np.nonzero(hold)[0][np.argmin(np.where(hold[hold], oarea[hold], np.inf))]
            host, depth = int(ok.node_idx.iloc[h]), int(ok.depth.iloc[h]) + 1
        else:
            host, depth = int(raw.node_idx.iloc[0]), int(raw.depth.iloc[0]) + 1
        row = {"screen_id": sid, "node_idx": INJECT_BASE + k, "parent_idx": host, "box_ok": True,
               **dict(zip(BOX, b)), "depth": depth, "n_children": int(donors["n_children"][j]),
               "class_name": donors["class_name"][j], **dict(zip(FLAGS, donors["flags"][j])),
               "label": 0, "is_labelled": True}
        rows.append(row)
    import pandas as pd
    new = pd.DataFrame(rows)[list(raw.columns)]
    return pd.concat([raw, new], ignore_index=True), [r["node_idx"] for r in rows]


def drop_raw(raw, keep, p, rng):
    """Remove each raw node not in `keep` with probability p (its children move up: raw_chain skips
    nodes without a usable box)."""
    if p <= 0:
        return raw
    raw = raw.copy()
    hit = (rng.random(len(raw)) < p) & ~raw.node_idx.isin(keep).to_numpy()
    raw.loc[hit, "box_ok"] = False
    return raw


def elements_as_leaves(chain, elems):
    """Elements are leaves of the input tree: a node whose raw parent is an element moves up to the
    nearest ancestor that is not one. Valid leaves never hold each other; about 9% of real invalid
    elements sit under another element in the raw tree (build check)."""
    out = {}
    for n, p in chain.items():
        while p != VROOT and p in elems:
            p = chain.get(p, VROOT)
        out[n] = p
    return out


# ---------------------------------------------------------------- one level of one screen
def build_level(g, raw, extra, pool_cfg, min_iou, masked=frozenset(), absent=False, kinds=None):
    """g: valid elements (node_idx, gt_parent, label, nx0..ny1). raw: raw nodes incl. injected rows
    (+ label, is_labelled). extra: element ids beyond the valid leaves (invalid + injected)."""
    true = dict(zip(g.node_idx.tolist(), g.gt_parent.tolist()))
    vleaves = set(true) - set(true.values())
    rw = raw.set_index("node_idx")
    # set built as Models 4-6 build their leaf set, so with no extra elements the pool matches their order too
    elems = vleaves if (absent or not extra) else vleaves | set(extra)
    ids = sorted(elems)                                             # node order, as in the candidate-pool check
    vbox = g.set_index("node_idx")[BOX]
    # boxes kept in the tables' float32 (as the candidate-pool check and Models 4-6): generated candidates and position ties match
    allb32 = np.array([vbox.loc[i].to_numpy() if i in vleaves else rw.loc[i, BOX].to_numpy() for i in ids],
                      np.float32).reshape(-1, 4)
    allb = allb32.astype(np.float64)
    gen = generated_candidates(ids, allb32, pool_cfg)               # node order, unclipped (as in the candidate-pool check)
    cent = (allb32[:, [0, 1]] + allb32[:, [2, 3]]) / 2
    order = np.lexsort((np.array(ids), cent[:, 0], cent[:, 1]))     # position (top, left), then id
    lids = [ids[i] for i in order]
    lbox = allb[order].clip(-0.5, 1.5)
    lcent = np.column_stack([(lbox[:, 0] + lbox[:, 2]) / 2, (lbox[:, 1] + lbox[:, 3]) / 2])
    vlab = dict(zip(g.node_idx, g.label))
    kind = np.array([0 if l in vleaves else (kinds or {}).get(l, 1) for l in lids], np.int8)
    msk = np.array([1.0 if (absent or l in masked) else 0.0 for l in lids])
    lmeta = np.column_stack([rw.loc[lids, "depth"].to_numpy(float), np.log1p(rw.loc[lids, "n_children"].to_numpy(float)),
                             rw.loc[lids, FLAGS].to_numpy(float)]) * (1 - msk[:, None])
    names = [("" if m else n) for n, m in zip(_short(rw.loc[lids, "class_name"]), msk)]
    out = {"screen_id": int(g.screen_id.iloc[0]), "leaf_ids": np.array(lids), "leaf_box": box_feats(lbox),
           "leaf_meta": np.column_stack([lmeta, msk]), "leaf_name": names,
           "leaf_label": np.array([vlab.get(l, 0) for l in lids], np.int64), "leaf_trust": (kind == 0).astype(np.float32),
           "leaf_kind": kind, "true": true, "leaf_xyxy": lbox, "has_meta": not absent}
    true_sets = [s for _, s in leaf_sets_of(true, vleaves)]

    chain = {}
    if absent:
        raw_sets = []
    else:
        chain = prune(elements_as_leaves(raw_chain(raw, elems), elems), elems)
        raw_sets = leaf_sets_of(chain, elems)
    raw_vals = {s for _, s in raw_sets}
    gen_extra = [(s, v) for s, v in gen.items() if s not in raw_vals]
    cids = [c for c, _ in raw_sets] + [("gen", k) for k in range(len(gen_extra))]
    sets = [s for _, s in raw_sets] + [s for s, _ in gen_extra]
    srcs = [gen.get(s, []) for _, s in raw_sets] + [v for _, v in gen_extra]
    member, ub, geo, src, st = _pool_arrays(sets, lids, lbox, lcent, srcs, src_names(pool_cfg))
    nr = len(raw_sets)
    rn = [c[1] for c, _ in raw_sets]
    rbox = rw.loc[rn, BOX].to_numpy(np.float64).clip(-0.5, 1.5).reshape(-1, 4)
    from uistruct.data.nesting_tokens import CAND_META
    meta = np.zeros((len(sets), len(CAND_META)))
    if nr:
        meta[:nr, 0] = 1
        meta[:nr, 1:7] = box_feats(rbox)
        meta[:nr, 7] = rw.loc[rn, "depth"].to_numpy(float)
        meta[:nr, 8] = np.log1p(rw.loc[rn, "n_children"].to_numpy(float))
        meta[:nr, 9:9 + len(FLAGS)] = rw.loc[rn, FLAGS].to_numpy(float)
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
        union = size[:, None] + size[None, :nr] - inter[:, :nr]
        meta[:, j0 + 2] = (inter[:, :nr] / np.maximum(union, 1)).max(1)
    cpos = {c: k for k, c in enumerate(cids[:nr])}
    leaf_rawpar = np.array([cpos.get(chain.get(l), -1) for l in lids])
    cand_rawpar = np.array([cpos.get(chain.get(c), -1) for c in cids[:nr]] + [-1] * len(gen_extra))
    proj = [s & vleaves for s in sets]                              # invalid members removed
    y_iou = oracle_scores(true_sets, proj, min_iou) if sets else np.zeros(0)
    y_iou[[i for i, s in enumerate(proj) if not s]] = 0
    lab = rw.loc[rn, "label"].to_numpy() if nr else np.zeros(0)
    isl = rw.loc[rn, "is_labelled"].to_numpy(bool) if nr else np.zeros(0, bool)
    ctrust = np.full(len(sets), -1.0, np.float32)
    ctrust[:nr] = np.where(isl & (lab >= 1), 1.0, np.where(isl & (lab == 0), 0.0, -1.0))
    out["M"] = {"sets": sets, "member": member, "geo": geo, "src": src, "struct": st, "meta": meta,
                "name": _short(rw.loc[rn, "class_name"]) + [""] * len(gen_extra),
                "leaf_rawpar": leaf_rawpar, "cand_rawpar": cand_rawpar,
                "y": (y_iou > 0).astype(np.float32), "y_iou": y_iou, "trust": ctrust,
                "crop_xyxy": np.vstack([rbox, ub[nr:]]) if nr else ub}
    return out


def level_settings(name, rates):
    """Corruption settings of a level (plan §2). rates: measured on train (p_inj)."""
    p = rates["p_inj"]
    return {"real": dict(p_inj=0, p_drop=0, p_mask=0, absent=False),
            "heavy": dict(p_inj=p, p_drop=0.25, p_mask=0.25, absent=False),
            "heavy_noinject": dict(p_inj=0, p_drop=0.25, p_mask=0.25, absent=False),   # fingerprint check
            "inject2": dict(p_inj=2 * p, p_drop=0, p_mask=0, absent=False),
            "drop50": dict(p_inj=0, p_drop=0.5, p_mask=0.5, absent=False),
            "absent": dict(p_inj=0, p_drop=0, p_mask=0, absent=True)}[name]


def build_screen_levels(g, raw, pool_cfg, min_iou, levels, rates, donors, seed):
    """All requested levels of one screen. The seed is per (screen, level), so a level never changes
    when another is added."""
    true = dict(zip(g.node_idx.tolist(), g.gt_parent.tolist()))
    vleaves = set(true) - set(true.values())
    inv = invalid_elements(raw, vleaves)
    sid = int(g.screen_id.iloc[0])
    out = {"screen_id": sid, "v": {}}
    for name in levels:
        s = level_settings(name, rates)
        rng = np.random.default_rng([seed, sid, LEVEL_ID[name]])
        r, fakes = raw, []
        if s["p_inj"]:
            r, fakes = inject(raw, int(rng.binomial(len(vleaves), s["p_inj"])), donors, rng, vleaves | set(inv))
        extra = inv + fakes
        r = drop_raw(r, set(vleaves) | set(extra), s["p_drop"], rng)
        elems = sorted(vleaves) + extra
        masked = frozenset(e for e in elems if rng.random() < s["p_mask"]) if s["p_mask"] else frozenset()
        kinds = {**{e: 1 for e in inv}, **{e: 2 for e in fakes}}
        out["v"][name] = build_level(g, r, extra, pool_cfg, min_iou, masked, s["absent"], kinds)
    return out
