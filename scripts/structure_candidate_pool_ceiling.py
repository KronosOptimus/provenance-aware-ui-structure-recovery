"""How good could the candidate-container nesting head get? Candidate-pool ceiling on clean dev.

For each pool, an oracle keeps exactly the candidates that match a true container (Hungarian on
leaf-set IoU, IoU >= min_iou), `assemble` turns them into a tree, and `grouping_metrics` scores it.
That is the best container F1 any scorer could reach from that pool with this assembler.

Pools (src/uistruct/models/nesting.py):
  P_xy, P_agglo, P_all, P_all_no_single   generated from leaf boxes only (pixels-only condition)
  M_raw                                   raw view-hierarchy nodes holding a given leaf
  M_all                                   M_raw + generated (metadata conditions)
Rule baselines are recomputed on the same screens: xycut (pixels only), raw_chain_collapsed (metadata).
Go/no-go (fixed in the config before the run): GO if ceiling(P_all) >= xycut + margin, raw truth.
Clean dev only. Run: python scripts/structure_candidate_pool_ceiling.py --config configs/structure_candidate_pool_ceiling.yaml
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from structure_rule_baselines import prune, raw_chain  # noqa: E402

from uistruct.eval.metrics import VROOT, density_bin, grouping_metrics, smooth_single_child  # noqa: E402
from uistruct.eval.stats import bootstrap_mean, paired_bootstrap  # noqa: E402
from uistruct.models.nesting import (assemble, best_iou_per_true, generated_candidates,  # noqa: E402
                                     leaf_sets_of, oracle_scores, xy_cut)
from uistruct.runlog import finish_run, start_run  # noqa: E402

BOX = ["nx0", "ny0", "nx1", "ny1"]
METRICS = ["container_f1", "container_match", "container_precision", "edge_f1", "leaf_edge_f1", "edit_distance"]
POOL_SOURCES = {
    "P_xy": lambda src: src in ("single", "all") or src.startswith("xy"),
    "P_agglo": lambda src: src in ("single", "all") or src.startswith("agglo"),
    "P_all": lambda src: True,
    "P_all_no_single": lambda src: src != "single",
}


def screen_rows(args):
    sid, g, raw, cfg = args
    true = dict(zip(g.node_idx, g.gt_parent))
    leaves = set(true) - set(true.values())
    lf = g[g.node_idx.isin(leaves)].sort_values("node_idx")
    lids, lb = list(lf.node_idx), lf[BOX].to_numpy()

    gen = generated_candidates(lids, lb, cfg["pool"])
    chain = prune(raw_chain(raw, leaves), leaves)
    raw_sets = leaf_sets_of(chain, leaves)
    raw_set_values = {s for _, s in raw_sets}
    pools = {}
    for name, keep in POOL_SOURCES.items():
        pools[name] = [(("gen", k), s) for k, (s, srcs) in enumerate(gen.items()) if any(keep(x) for x in srcs)]
    pools["M_raw"] = raw_sets
    pools["M_all"] = raw_sets + [(("gen", k), s) for k, s in enumerate(gen) if s not in raw_set_values]

    tops, edges = xy_cut(lids, lb, cfg["contain_tol"])
    rules = {"xycut": {**edges, **{t: VROOT for t in tops}},
             "raw_chain_collapsed": smooth_single_child(chain, leaves)}

    rows = []
    base = {"screen_id": sid, "bin": density_bin(len(leaves)), "n_leaves": len(leaves)}
    for tname, t in {"raw": true, "smoothed": smooth_single_child(true)}.items():
        true_sets = [s for _, s in leaf_sets_of(t, leaves)]
        for name, p in rules.items():
            rows.append({**base, "truth": tname, "system": name, "pool_size": np.nan, **grouping_metrics(t, p)})
        for name, pool in pools.items():
            sets = [s for _, s in pool]
            sc = oracle_scores(true_sets, sets, cfg["min_iou"])
            pred = assemble([(cid, s, x) for (cid, s), x in zip(pool, sc)], leaves, threshold=cfg["min_iou"])
            best = best_iou_per_true(true_sets, sets)
            rows.append({**base, "truth": tname, "system": f"ceiling_{name}", "pool_size": len(pool),
                         "cover50": float((best >= 0.5).mean()) if len(best) else np.nan,
                         "cover90": float((best >= 0.9).mean()) if len(best) else np.nan,
                         "cover100": float((best >= 1.0).mean()) if len(best) else np.nan,
                         **grouping_metrics(t, pred)})
    return rows


def main(cfg, run_dir):
    el = pd.read_parquet(f"{cfg['data_dir']}/elements.parquet")
    el = el[el.is_valid & (el.split_clean == cfg["split"])]
    nodes = pd.read_parquet(f"{cfg['data_dir']}/nodes.parquet", columns=["screen_id", "node_idx", "parent_idx", "box_ok", *BOX])
    nodes = nodes[nodes.screen_id.isin(set(el.screen_id))]
    raw_by = {k: v.sort_values("node_idx") for k, v in nodes.groupby("screen_id", sort=False)}
    jobs = [(sid, g[["node_idx", "gt_parent", *BOX]], raw_by[sid], cfg) for sid, g in el.groupby("screen_id", sort=True)]
    if cfg.get("limit"):
        jobs = jobs[: cfg["limit"]]
    del nodes, raw_by, el                     # free memory before the workers start
    rows = []
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        for i, r in enumerate(ex.map(screen_rows, jobs, chunksize=32)):
            rows.extend(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)} screens", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "per_screen.csv", index=False)

    out = {"split": cfg["split"], "n_screens": int(df.screen_id.nunique())}
    for (tname, name), x in df.groupby(["truth", "system"]):
        r = {m: bootstrap_mean(x[m], cfg["n_boot"], cfg["seed"]) for m in METRICS}
        r["by_density_container_f1"] = {b: round(float(v.container_f1.mean()), 4) for b, v in x.groupby("bin")}
        r["mean_pred_containers"] = float(x.n_pred_containers.mean())
        r["mean_true_containers"] = float(x.n_true_containers.mean())
        if name.startswith("ceiling_"):
            r["mean_pool_size"] = float(x.pool_size.mean())
            for c in ["cover50", "cover90", "cover100"]:
                r[c] = float(x[c].mean())
        out[f"{name}|{tname}"] = r
    for tname in ["raw", "smoothed"]:
        d = df[df.truth == tname].pivot(index="screen_id", columns="system", values="container_f1")
        out[f"ceiling_P_all_minus_xycut|{tname}"] = paired_bootstrap(d["ceiling_P_all"], d["xycut"], cfg["n_boot"], cfg["seed"])
        out[f"ceiling_M_all_minus_raw_chain_collapsed|{tname}"] = paired_bootstrap(
            d["ceiling_M_all"], d["raw_chain_collapsed"], cfg["n_boot"], cfg["seed"])
        out[f"ceiling_M_all_minus_ceiling_P_all|{tname}"] = paired_bootstrap(
            d["ceiling_M_all"], d["ceiling_P_all"], cfg["n_boot"], cfg["seed"])

    ceil = out[f"ceiling_{cfg['gate']['pool']}|raw"]["container_f1"]["mean"]
    xy = out["xycut|raw"]["container_f1"]["mean"]
    out["gate"] = {"pool": cfg["gate"]["pool"], "ceiling_container_f1": ceil, "xycut_container_f1": xy,
                   "margin": cfg["gate"]["margin"], "needed": xy + cfg["gate"]["margin"],
                   "decision": "GO" if ceil >= xy + cfg["gate"]["margin"] else "NO-GO: add pairwise proposals"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N screens only (smoke test; not a result)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.limit:
        cfg["limit"] = a.limit
    run_dir = start_run("structure_candidate_pool" + ("_smoke" if a.limit else ""), cfg)
    res = main(cfg, run_dir)
    finish_run(run_dir, res)
    print(f"{'system|truth':40s} {'contF1':>7s} {'CM':>6s} {'prec':>6s} {'edgeF1':>6s} {'pred/true':>10s} {'pool':>6s} {'cov50':>6s} {'cov90':>6s}")
    for k, v in res.items():
        if isinstance(v, dict) and "container_f1" in v:
            print(f"{k:40s} {v['container_f1']['mean']:7.3f} {v['container_match']['mean']:6.3f} "
                  f"{v['container_precision']['mean']:6.3f} {v['edge_f1']['mean']:6.3f} "
                  f"{v['mean_pred_containers']:4.1f}/{v['mean_true_containers']:4.1f} "
                  f"{v.get('mean_pool_size', float('nan')):6.1f} {v.get('cover50', float('nan')):6.3f} {v.get('cover90', float('nan')):6.3f}")
        elif isinstance(v, dict) and "diff" in v:
            print(f"{k:55s} diff={v['diff']:.3f} [{v['lo']:.3f},{v['hi']:.3f}] p={v['p_a_not_better']:.3f}")
    print("gate:", res["gate"])
    print("run dir:", run_dir)
