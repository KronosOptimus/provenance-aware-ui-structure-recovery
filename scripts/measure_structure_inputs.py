"""Measurements of what the nesting step receives.

All on clean DEV (test is kept for final runs only). Four measurements:
  structure   how many valid elements are containers, of which types; single-child containers;
              does a container's box equal the box around its leaves (then it carries no extra
              information beyond the grouping)?
  pixels      are container borders visible in the screenshot? Proxy: mean absolute difference
              between pixel strips just inside and just outside each box edge, for containers,
              leaves, and random boxes of the same size (control).
  leaves_only given only the true leaf boxes, how well do non-learned rules build the tree?
              flat, and recursive XY-cut (split on whitespace gaps, alternating axes); scored with
              grouping_metrics (leaf-set container matching), on the raw and smoothed true tree.
              For reference, smallest-box over all valid boxes (containers given) is scored the same way.
  metadata    all CLAY-labelled boxes, invalid ones included, offered to the smallest-box rule:
              share of valid elements whose nearest valid ancestor is recovered (scripts the 95.58%
              earlier scratch check, here on dev).
Run: python scripts/measure_structure_inputs.py --config configs/measure_structure_inputs.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from structure_rule_baselines_true_elements import smallest_container  # noqa: E402

from uistruct.eval.metrics import VROOT, density_bin, grouping_metrics, smooth_single_child  # noqa: E402
from uistruct.eval.stats import bootstrap_mean  # noqa: E402
from uistruct.runlog import finish_run, start_run  # noqa: E402

LABELS = {int(k): v.strip() for k, v in (l.split(":") for l in open("data/raw/clay/label_map.txt"))}
METRICS = ["edge_f1", "leaf_edge_f1", "container_match", "edit_distance"]


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------- structure
def structure(el):
    rows, iou_rows = [], []
    for sid, g in el.groupby("screen_id", sort=False):
        ids = g.node_idx.to_numpy()
        par = dict(zip(ids, g.gt_parent.to_numpy()))
        box = dict(zip(ids, g[["nx0", "ny0", "nx1", "ny1"]].to_numpy()))
        lab = dict(zip(ids, g.label.to_numpy()))
        ch = {}
        for c, p in par.items():
            ch.setdefault(p, []).append(c)
        leaves = set(ids) - set(ch)

        def leafs(n):
            return [n] if n in leaves else [x for c in ch[n] for x in leafs(c)]

        for n in ids:
            if n in ch:
                ls = leafs(n)
                lb = np.array([box[x] for x in ls])
                u = (lb[:, 0].min(), lb[:, 1].min(), lb[:, 2].max(), lb[:, 3].max())
                single = len(ch[n]) == 1
                iou_rows.append({"label": lab[n], "n_children": len(ch[n]), "single": single,
                                 "single_same_box": single and box_iou(box[n], box[ch[n][0]]) > 0.9,
                                 "iou_leaf_union": box_iou(box[n], u)})
            rows.append({"label": lab[n], "is_container": n in ch})
    r, c = pd.DataFrame(rows), pd.DataFrame(iou_rows)
    by_type = r.groupby("label").is_container.agg(["mean", "sum", "count"])
    share = c.label.value_counts(normalize=True)
    return {
        "n_valid_elements": len(r), "n_containers": len(c),
        "container_share": float(r.is_container.mean()),
        "containers_by_type": {LABELS[int(k)]: round(float(v), 4) for k, v in share.head(10).items()},
        "share_of_type_that_is_container": {LABELS[int(k)]: round(float(v["mean"]), 4)
                                            for k, v in by_type.iterrows() if v["count"] >= 500},
        "single_child_share": float(c.single.mean()),
        "single_child_same_box_share": float(c.single_same_box.mean()),
        "box_vs_leaf_union_iou_median": float(c.iou_leaf_union.median()),
        "box_vs_leaf_union_iou_ge_0.95": float((c.iou_leaf_union >= 0.95).mean()),
        "box_vs_leaf_union_iou_lt_0.8": float((c.iou_leaf_union < 0.8).mean()),
    }


# ---------------------------------------------------------------- pixels
def edge_contrast(img, b, w):
    """Mean |inside strip - outside strip| per edge (0-255), NaN for edges at the screen border."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(round(b[0] * W)), int(round(b[1] * H)), int(round(b[2] * W)), int(round(b[3] * H)))
    out = []
    if x1 - x0 < 2 * w + 2 or y1 - y0 < 2 * w + 2:
        return [np.nan] * 4
    ya, yb = max(y0, 0), min(y1, H)
    xa, xb = max(x0, 0), min(x1, W)
    for strips in [
        ((slice(ya, yb), slice(x0, x0 + w)), (slice(ya, yb), slice(x0 - w, x0))) if x0 - w >= 0 else None,
        ((slice(ya, yb), slice(x1 - w, x1)), (slice(ya, yb), slice(x1, x1 + w))) if x1 + w <= W else None,
        ((slice(y0, y0 + w), slice(xa, xb)), (slice(y0 - w, y0), slice(xa, xb))) if y0 - w >= 0 else None,
        ((slice(y1 - w, y1), slice(xa, xb)), (slice(y1, y1 + w), slice(xa, xb))) if y1 + w <= H else None,
    ]:
        if strips is None or yb <= ya or xb <= xa:
            out.append(np.nan)
            continue
        inside, outside = strips
        a, o = img[inside[0], inside[1]], img[outside[0], outside[1]]
        if a.shape != o.shape or a.size == 0:
            out.append(np.nan)
            continue
        out.append(float(np.abs(a - o).mean()))
    return out


def pixels(el, cfg, rng):
    sids = rng.choice(el.screen_id.unique(), cfg["pixel_sample_screens"], replace=False)
    rows = []
    for sid in sids:
        g = el[el.screen_id == sid]
        img = np.asarray(Image.open(f"{cfg['image_dir']}/{sid}.jpg").convert("RGB"), dtype=np.float32)
        is_cont = set(g.gt_parent.to_numpy())
        for n, lab, *b in g[["node_idx", "label", "nx0", "ny0", "nx1", "ny1"]].itertuples(index=False):
            kind = "container" if n in is_cont else "leaf"
            rows.append({"kind": kind, "label": lab, "edges": edge_contrast(img, b, cfg["strip_px"])})
            bw, bh = b[2] - b[0], b[3] - b[1]   # control: same size, random position on the same screen
            if bw < 1 and bh < 1:
                rx, ry = rng.uniform(0, 1 - bw), rng.uniform(0, 1 - bh)
                rows.append({"kind": "random_" + kind, "label": lab,
                             "edges": edge_contrast(img, [rx, ry, rx + bw, ry + bh], cfg["strip_px"])})
    out = {}
    t = cfg["edge_threshold"]
    for kind in ["container", "leaf", "random_container", "random_leaf"]:
        e = [r["edges"] for r in rows if r["kind"] == kind]
        arr = np.array(e, float)
        usable = ~np.isnan(arr)
        vis = (arr > t) & usable
        n_use = usable.sum(1)
        ok = n_use >= 2
        out[kind] = {"n_boxes": int(ok.sum()),
                     "edge_visible_rate": float(vis.sum() / usable.sum()),
                     "all_usable_edges_visible": float((vis.sum(1)[ok] == n_use[ok]).mean()),
                     "no_edge_visible": float((vis.sum(1)[ok] == 0).mean()),
                     "median_edge_contrast": float(np.nanmedian(arr))}
    per_type = {}
    for lab in sorted({r["label"] for r in rows if r["kind"] == "container"}):
        arr = np.array([r["edges"] for r in rows if r["kind"] == "container" and r["label"] == lab], float)
        usable = ~np.isnan(arr)
        ok = usable.sum(1) >= 2
        if ok.sum() >= 30:
            vis = (arr > t) & usable
            per_type[LABELS[int(lab)]] = {"n": int(ok.sum()),
                                          "all_usable_edges_visible": round(float((vis.sum(1)[ok] == usable.sum(1)[ok]).mean()), 4),
                                          "no_edge_visible": round(float((vis.sum(1)[ok] == 0).mean()), 4)}
    out["container_by_type"] = per_type
    out["n_screens"] = int(len(sids))
    return out


# ---------------------------------------------------------------- leaves only
def xy_cut(ids, b, tol, depth=0, counter=None):
    """Recursive XY-cut over leaf boxes. Returns (children, edges): the top-level nodes of this group
    and {child: parent} edges inside it. New containers get ids ('xy', k)."""
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
        return list(ids), {}          # no whitespace cut in either direction: flat group
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


def leaves_only(el, cfg):
    rows = []
    for sid, g in el.groupby("screen_id", sort=True):
        ids = g.node_idx.to_numpy()
        true = dict(zip(ids, g.gt_parent.to_numpy()))
        is_par = set(true.values())
        leaf = g[~g.node_idx.isin(is_par)]
        lids = list(leaf.node_idx.to_numpy())
        lb = leaf[["nx0", "ny0", "nx1", "ny1"]].to_numpy()
        tops, edges = xy_cut(lids, lb, cfg["contain_tol"])
        xy = {**edges, **{t: VROOT for t in tops}}
        preds = {"flat_leaves": {i: VROOT for i in lids},
                 "xycut_leaves": xy,
                 "smallest_all_valid_boxes": smallest_container(g[["nx0", "ny0", "nx1", "ny1"]].to_numpy(), ids,
                                                                cfg["contain_tol"])}
        truths = {"raw": true, "smoothed": smooth_single_child(true)}
        for tname, t in truths.items():
            for name, p in preds.items():
                if name == "smallest_all_valid_boxes" and tname == "smoothed":
                    continue  # its containers are the given boxes; scored against the raw tree only
                m = grouping_metrics(t, p)
                rows.append({"screen_id": sid, "truth": tname, "baseline": name,
                             "bin": density_bin(len(lids)), "n_leaves": len(lids), **m})
    df = pd.DataFrame(rows)
    out = {}
    for (tname, name), x in df.groupby(["truth", "baseline"]):
        out[f"{name}|{tname}"] = {m: bootstrap_mean(x[m], cfg["n_boot"], cfg["seed"]) for m in METRICS}
        out[f"{name}|{tname}"]["by_density_cm"] = {b: round(float(v.container_match.mean()), 4)
                                                   for b, v in x.groupby("bin")}
    return out, df


# ---------------------------------------------------------------- metadata (invalid boxes offered)
def with_invalid(el_all, cfg):
    right, total = 0, 0
    for sid, g in el_all.groupby("screen_id", sort=True):
        ids = g.node_idx.to_numpy()
        p = smallest_container(g[["nx0", "ny0", "nx1", "ny1"]].to_numpy(), ids, cfg["contain_tol"])
        valid = dict(zip(ids, g.is_valid.to_numpy()))
        for n, gt in zip(ids, g.gt_parent.to_numpy()):
            if not valid[n]:
                continue
            a = p[n]
            while a != VROOT and not valid[a]:  # skip offered invalid boxes, as the true tree does
                a = p[a]
            right += a == gt
            total += 1
    return {"valid_elements": total, "nearest_valid_ancestor_right": right / total}


def main(cfg, run_dir):
    rng = np.random.default_rng(cfg["seed"])
    el_all = pd.read_parquet(f"{cfg['data_dir']}/elements.parquet")
    el_all = el_all[el_all.split_clean == cfg["split"]]
    el = el_all[el_all.is_valid]
    res = {"split": cfg["split"], "n_screens": int(el.screen_id.nunique())}
    res["structure"] = structure(el)
    print("structure done", flush=True)
    res["pixels"] = pixels(el, cfg, rng)
    print("pixels done", flush=True)
    res["leaves_only"], df = leaves_only(el, cfg)
    df.to_csv(run_dir / "leaves_only_per_screen.csv", index=False)
    print("leaves_only done", flush=True)
    res["metadata_with_invalid"] = with_invalid(el_all, cfg)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    run_dir = start_run("structure_inputs", cfg)
    res = main(cfg, run_dir)
    finish_run(run_dir, res)
    import json
    print(json.dumps(res, indent=1, default=float))
    print("run dir:", run_dir)
