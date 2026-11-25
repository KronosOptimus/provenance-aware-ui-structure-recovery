"""Non-learned nesting baselines for every metadata condition, one table, same scoring.

Task: the true leaf elements are given; containers
must be created. Scored with grouping_metrics on the raw and the smoothed true tree; container
precision and container F1 added here (CM cannot see extra containers).

Pixels-only condition (leaf boxes only):
  flat                 every leaf hangs from the virtual root (floor)
  xycut                recursive XY-cut over the leaf boxes (as in the nesting-input measurements)
Metadata condition: every raw RICO node of the screen with a usable box (`box_ok`) is a candidate
container. CLAY-labelled nodes are NOT singled out: that would leak CLAY's labelling choice. Raw
candidates get ids ("raw", node_idx), so they are matched to true containers only by their leaf sets.
  raw_chain            the raw view hierarchy itself: each leaf keeps its raw ancestors (nodes without a
                       usable box are skipped); nodes with no given leaf below them are pruned
  raw_chain_collapsed  same, with single-child containers removed (wrapper chains collapse)
  raw_smallest_box     parent = smallest raw box containing the node (metadata boxes, hierarchy links
                       ignored), then pruned and collapsed as above
Reference (not a baseline: leaks the true container boxes):
  given_container_boxes smallest box over all valid boxes, containers included, raw truth only
Clean dev only. Run: python scripts/structure_rule_baselines.py --config configs/structure_rule_baselines.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from structure_rule_baselines_true_elements import smallest_container  # noqa: E402
from measure_structure_inputs import xy_cut  # noqa: E402

from uistruct.eval.metrics import VROOT, density_bin, grouping_metrics, smooth_single_child  # noqa: E402
from uistruct.eval.stats import bootstrap_mean, paired_bootstrap  # noqa: E402
from uistruct.runlog import finish_run, start_run  # noqa: E402

METRICS = ["edge_f1", "leaf_edge_f1", "container_match", "container_precision", "container_f1", "edit_distance"]
BOX = ["nx0", "ny0", "nx1", "ny1"]


def prune(parents: dict, leaves: set) -> dict:
    """Keep the given leaves and their ancestors; re-id non-leaf nodes as ("raw", idx)."""
    keep = set()
    for l in leaves:
        n = l
        while n != VROOT and n not in keep:
            keep.add(n)
            n = parents[n]
    name = lambda n: n if (n in leaves or n == VROOT) else ("raw", n)  # noqa: E731
    return {name(n): name(parents[n]) for n in keep}


def raw_chain(raw: pd.DataFrame, leaves: set) -> dict:
    """Raw hierarchy with unusable-box nodes skipped (their children move up)."""
    par = dict(zip(raw.node_idx, raw.parent_idx))
    ok = dict(zip(raw.node_idx, raw.box_ok))
    out = {}
    for n in raw.node_idx:
        if not ok[n] and n not in leaves:
            continue
        p = par[n]
        while p != VROOT and not ok[p]:
            p = par[p]
        out[n] = p
    return out


def predictions(g, raw, leaves, tol):
    lf = g[g.node_idx.isin(leaves)]
    tops, edges = xy_cut(list(lf.node_idx), lf[BOX].to_numpy(), tol)
    chain = prune(raw_chain(raw, leaves), leaves)
    cand = raw[raw.box_ok | raw.node_idx.isin(leaves)]
    sb = prune(smallest_container(cand[BOX].to_numpy(), cand.node_idx.to_numpy(), tol), leaves)
    return {"flat": {i: VROOT for i in leaves},
            "xycut": {**edges, **{t: VROOT for t in tops}},
            "raw_chain": chain,
            "raw_chain_collapsed": smooth_single_child(chain, leaves),
            "raw_smallest_box": smooth_single_child(sb, leaves),
            "given_container_boxes": smallest_container(g[BOX].to_numpy(), g.node_idx.to_numpy(), tol)}


def main(cfg, run_dir):
    el = pd.read_parquet(f"{cfg['data_dir']}/elements.parquet")
    el = el[el.is_valid & (el.split_clean == cfg["split"])]
    nodes = pd.read_parquet(f"{cfg['data_dir']}/nodes.parquet", columns=["screen_id", "node_idx", "parent_idx", "box_ok", *BOX])
    nodes = nodes[nodes.screen_id.isin(set(el.screen_id))]
    raw_by = dict(tuple(nodes.groupby("screen_id", sort=False)))
    rows = []
    for sid, g in el.groupby("screen_id", sort=True):
        true = dict(zip(g.node_idx, g.gt_parent))
        leaves = set(true) - set(true.values())
        preds = predictions(g, raw_by[sid].sort_values("node_idx"), leaves, cfg["contain_tol"])
        for tname, t in {"raw": true, "smoothed": smooth_single_child(true)}.items():
            for name, p in preds.items():
                if name == "given_container_boxes" and tname == "smoothed":
                    continue
                rows.append({"screen_id": sid, "truth": tname, "baseline": name, "bin": density_bin(len(leaves)),
                             "n_leaves": len(leaves), "n_raw_nodes": len(raw_by[sid]), **grouping_metrics(t, p)})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "per_screen.csv", index=False)
    out = {"split": cfg["split"], "n_screens": int(df.screen_id.nunique())}
    for (tname, name), x in df.groupby(["truth", "baseline"]):
        r = {m: bootstrap_mean(x[m], cfg["n_boot"], cfg["seed"]) for m in METRICS}
        r["by_density_cm"] = {b: round(float(v.container_match.mean()), 4) for b, v in x.groupby("bin")}
        r["by_density_container_f1"] = {b: round(float(v.container_f1.mean()), 4) for b, v in x.groupby("bin")}
        r["mean_pred_containers"] = float(x.n_pred_containers.mean())
        r["mean_true_containers"] = float(x.n_true_containers.mean())
        out[f"{name}|{tname}"] = r
    # metadata vs pixels-only, paired on the same screens
    for tname in ["raw", "smoothed"]:
        d = df[df.truth == tname].pivot(index="screen_id", columns="baseline", values="container_f1")
        for name in ["raw_chain", "raw_chain_collapsed", "raw_smallest_box"]:
            out[f"{name}_minus_xycut_container_f1|{tname}"] = paired_bootstrap(d[name], d["xycut"], cfg["n_boot"], cfg["seed"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    run_dir = start_run("structure_rule_baselines", cfg)
    res = main(cfg, run_dir)
    finish_run(run_dir, res)
    for k, v in res.items():
        if isinstance(v, dict) and "edge_f1" in v:
            print(f"{k:38s}", " ".join(f"{m}={v[m]['mean']:.3f}" for m in METRICS), f"pred/true={v['mean_pred_containers']:.1f}/{v['mean_true_containers']:.1f}")
        elif isinstance(v, dict):
            print(f"{k:50s} diff={v['diff']:.3f} [{v['lo']:.3f},{v['hi']:.3f}] p={v['p_a_not_better']:.3f}")
    print("run dir:", run_dir)
