"""Non-learned nesting baselines, which also exercise the whole evaluation pipeline.

  flat        every element hangs from the virtual root (score floor; container match = 0)
  smallest    parent = the smallest box that contains the element (geometry only, no metadata)

Inputs are the true valid elements (oracle boxes). No view-hierarchy order is used: ties between
identical boxes are broken by position order (top, left, larger first), so the rule is fair to a
metadata-free setting. Run: python scripts/structure_rule_baselines_true_elements.py --config configs/structure_rule_baselines_true_elements.yaml
"""
import argparse

import numpy as np
import pandas as pd
import yaml

from uistruct.eval.metrics import VROOT, density_bin, tree_metrics
from uistruct.eval.stats import bootstrap_mean, paired_bootstrap
from uistruct.runlog import finish_run, start_run

METRICS = ["edge_f1", "leaf_edge_f1", "container_match", "edit_distance"]


def smallest_container(b, ids, tol):
    """b: (n,4) boxes x0,y0,x1,y1. Returns {id: parent id or VROOT}."""
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    order = np.lexsort((-area, b[:, 0], b[:, 1]))  # position order: top, left, larger first
    rank = np.empty(len(b), int)
    rank[order] = np.arange(len(b))
    inside = ((b[None, :, 0] >= b[:, None, 0] - tol) & (b[None, :, 1] >= b[:, None, 1] - tol) &
              (b[None, :, 2] <= b[:, None, 2] + tol) & (b[None, :, 3] <= b[:, None, 3] + tol))  # [p, c]
    # p can be parent of c only if p is strictly "bigger" in the total order (area, then position rank)
    bigger = (area[:, None] > area[None, :]) | ((area[:, None] == area[None, :]) & (rank[:, None] < rank[None, :]))
    cand = inside & bigger
    out = {}
    for c in range(len(b)):
        ps = np.flatnonzero(cand[:, c])
        if len(ps) == 0:
            out[ids[c]] = VROOT
        else:  # smallest area; among equal areas the latest in position order (deepest)
            best = ps[np.lexsort((-rank[ps], area[ps]))[0]]
            out[ids[c]] = ids[best]
    return out


def main(cfg, run_dir):
    el = pd.read_parquet(f"{cfg['data_dir']}/elements.parquet")
    el = el[el.is_valid & el.split_clean.isin(cfg["splits"])]
    rows = []
    for (split, sid), g in el.groupby(["split_clean", "screen_id"], sort=True):
        ids = g.node_idx.to_numpy()
        true = dict(zip(ids, g.gt_parent.to_numpy()))
        preds = {"flat": {i: VROOT for i in ids},
                 "smallest": smallest_container(g[["nx0", "ny0", "nx1", "ny1"]].to_numpy(), ids, cfg["contain_tol"])}
        for name, p in preds.items():
            m = tree_metrics(true, p)
            rows.append({"split": split, "screen_id": sid, "baseline": name, "bin": density_bin(len(ids)), **m})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "per_screen.csv", index=False)
    out = {}
    for split in cfg["splits"]:
        d = df[df.split == split]
        out[split] = {}
        for name in ["flat", "smallest"]:
            x = d[d.baseline == name]
            out[split][name] = {m: bootstrap_mean(x[m], cfg["n_boot"], cfg["seed"]) for m in METRICS}
            out[split][name]["by_density"] = {b: {m: round(float(v[m].mean()), 4) for m in METRICS} | {"n": len(v)}
                                              for b, v in x.groupby("bin")}
        a = d[d.baseline == "smallest"].sort_values("screen_id")
        f = d[d.baseline == "flat"].sort_values("screen_id")
        out[split]["smallest_minus_flat_edge_f1"] = paired_bootstrap(a.edge_f1, f.edge_f1, cfg["n_boot"], cfg["seed"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    run_dir = start_run("structure_rule_baselines_true_elements", cfg)
    res = main(cfg, run_dir)
    finish_run(run_dir, res)
    for s in cfg["splits"]:
        for n in ["flat", "smallest"]:
            print(s, n, {m: f"{res[s][n][m]['mean']:.4f} [{res[s][n][m]['lo']:.4f}, {res[s][n][m]['hi']:.4f}]" for m in METRICS})
        print(s, "by density (smallest):", res[s]["smallest"]["by_density"])
    print("run dir:", run_dir)
