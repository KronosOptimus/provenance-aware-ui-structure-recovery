"""Analysis of saved outputs (no training, no selection, nothing new is run on the test split).

Stage `rescore`: container F1 under one rule for screens without true containers (configs/analyse_saved_outputs.yaml).
  The scoring code (eval.metrics.grouping_metrics) leaves container F1 undefined (NaN) when a screen has
  no true and no predicted container, and scores 0 when it has no true container but predicted ones.
  Systems were therefore averaged over different screens. Here both-empty screens score 1.0, so every
  system is averaged over all screens; a second version keeps only screens with >= 1 true container.
  The stored numbers are recomputed first and must match exactly.
Stage `leaf_types`: type macro-F1 over the 23 types of Models 1-6 on the same valid test leaves. Models 1-3
  rows are re-aligned to (screen_id, node_idx) by rebuilding their row order (features.load_elements order).

Usage: python scripts/analyse_saved_outputs.py rescore | leaf_types
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uistruct.eval.stats import bootstrap_mean, paired_bootstrap
from uistruct.runlog import finish_run, start_run
from uistruct.train.type_head import per_screen_confusion, scores as type_scores

RUNS = Path("results/runs")
TYPE_NAMES = {1: "IMAGE", 2: "PICTOGRAM", 3: "BUTTON", 4: "TEXT", 5: "LABEL", 6: "TEXT_INPUT", 7: "MAP",
              8: "CHECK_BOX", 9: "SWITCH", 10: "PAGER_INDICATOR", 11: "SLIDER", 12: "RADIO_BUTTON",
              13: "SPINNER", 14: "PROGRESS_BAR", 15: "ADVERTISEMENT", 16: "DRAWER", 17: "NAVIGATION_BAR",
              18: "TOOLBAR", 19: "LIST_ITEM", 20: "CARD_VIEW", 21: "CONTAINER", 22: "DATE_PICKER",
              23: "NUMBER_STEPPER"}


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


# ------------------------------------------------------------------ rescore
def rule_values(g, rule):
    """Per-screen container F1 of one system under a rule for screens without true containers."""
    f1 = g.container_f1.to_numpy(float).copy()
    empty_true = g.n_true_containers.to_numpy() == 0
    both_empty = empty_true & (g.n_pred_containers.to_numpy() == 0)
    if rule == "current":
        return f1
    if rule == "empty_is_1":
        assert np.isnan(f1[both_empty]).all() and not np.isnan(f1[~both_empty]).any()
        f1[both_empty] = 1.0
        return f1
    if rule == "true_only":
        f1[empty_true] = np.nan
        return f1
    raise ValueError(rule)


def stage_rescore(cfg, run_dir):
    pairs = yaml.safe_load((RUNS / cfg["pairs_from"] / "config.yaml").read_text())["pairs"]
    out, checks = {}, []
    for run in cfg["rescore_runs"]:
        ps = pd.read_csv(RUNS / run / "per_screen.csv")
        col = "system" if "system" in ps else "baseline"
        stored = json.loads((RUNS / run / "metrics.json").read_text())
        stored = stored.get("systems", stored)
        res = {"n_screens": int(ps.screen_id.nunique()), "systems": {}, "paired": {}}
        for truth in ["raw", "smoothed"]:
            pt = ps[ps.truth == truth]
            res[f"screens_without_true_container|{truth}"] = int(
                pt.groupby("screen_id").n_true_containers.first().eq(0).sum())
            wide = {}
            for name, g in pt.groupby(col):
                g = g.sort_values("screen_id")
                key = f"{name}|{truth}"
                r = {"n_pred_on_empty_screens": int(((g.n_true_containers == 0) & (g.n_pred_containers > 0)).sum())}
                for rule in ["current", "empty_is_1", "true_only"]:
                    v = rule_values(g, rule)
                    r[rule] = {k: r4(x) if k != "n" else x for k, x in
                               bootstrap_mean(v, cfg["n_boot"], cfg["seed"]).items()}
                    wide[(rule, name)] = pd.Series(v, index=g.screen_id.to_numpy())
                b = g.assign(v=rule_values(g, "empty_is_1")).groupby("bin").v.mean()
                r["by_density_empty_is_1"] = {k: r4(x) for k, x in b.items()}
                if key in stored:                               # reproduction check of the stored mean
                    old = stored[key]["container_f1"]["mean"]
                    checks.append((run, key, abs(old - bootstrap_mean(rule_values(g, "current"))["mean"])))
                res["systems"][key] = r
            names = {n for _, n in wide}
            for a, b in pairs:
                if a in names and b in names:
                    for rule in ["current", "empty_is_1", "true_only"]:
                        x, y = wide[(rule, a)].align(wide[(rule, b)], join="outer")
                        res["paired"][f"{a}_minus_{b}|{truth}|{rule}"] = {
                            k: r4(v) if k != "n" else v for k, v in
                            paired_bootstrap(x.to_numpy(), y.to_numpy(), cfg["n_boot"], cfg["seed"]).items()}
        out[run] = res
    worst = max(c[2] for c in checks)
    assert worst < 1e-12, f"stored means not reproduced (max diff {worst})"
    out["reproduction_check"] = {"n_compared": len(checks), "max_abs_diff": worst}
    return out


def print_rescore(m):
    print(f"reproduction check: {m['reproduction_check']}")
    for run, res in m.items():
        if run == "reproduction_check":
            continue
        print(f"\n== {run}  screens {res['n_screens']}  without true container (raw) "
              f"{res['screens_without_true_container|raw']}")
        print(f"{'system|truth':32s} {'current':>8s} {'n':>5s} {'empty=1':>8s} {'[95% CI]':>17s} {'true>=1':>8s}")
        for k, r in res["systems"].items():
            if k.endswith("|raw"):
                e = r["empty_is_1"]
                print(f"{k:32s} {r['current']['mean']:8.4f} {r['current']['n']:5d} {e['mean']:8.4f} "
                      f"[{e['lo']:.4f}, {e['hi']:.4f}] {r['true_only']['mean']:8.4f}")
        for k, p in res["paired"].items():
            if "|raw|" in k:
                print(f"  {k:52s} {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] p={p['p_a_not_better']:.3f} n={p['n']}")


# ------------------------------------------------------------------ leaf types
def type_row_order(split="test", data_dir="data/processed"):
    """(screen_id, node_idx, label) in the row order of features.load_elements (Models 1-3 outputs).
    Same steps as load_elements, with only the columns the order depends on (laptop RAM)."""
    el = pd.read_parquet(f"{data_dir}/elements.parquet",
                         columns=["screen_id", "node_idx", "label", "is_valid", "split_clean"])
    el = el[el.split_clean.isin((split,))]
    nodes = pd.read_parquet(f"{data_dir}/nodes.parquet", columns=["screen_id", "node_idx", "nx0", "ny0", "nx1", "ny1"],
                            filters=[("screen_id", "in", sorted(el.screen_id.unique().tolist()))])
    df = el.merge(nodes, on=["screen_id", "node_idx"], how="left", validate="one_to_one")
    b = df[["nx0", "ny0", "nx1", "ny1"]].to_numpy(np.float64).clip(-0.5, 1.5)
    df["cx"], df["cy"] = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    df = df.sort_values(["screen_id", "cy", "cx"]).reset_index(drop=True)
    ties = int(df.duplicated(["screen_id", "cy", "cx"]).sum())
    return df[["screen_id", "node_idx", "label"]], ties


def fast_boot(x, stat, n_boot, seed, y=None):
    """bootstrap_stat / paired_bootstrap_stat with the same random draws, summing by resample counts."""
    rng = np.random.default_rng(seed)
    n = len(x)
    x2 = x.reshape(n, -1).astype(np.int64)
    y2 = None if y is None else y.reshape(n, -1).astype(np.int64)
    shape = x.shape[1:]
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        v = stat((w @ x2).reshape(shape))
        if y2 is not None:
            v -= stat((w @ y2).reshape(shape))
        boots.append(v)
    boots = np.array(boots)
    val = stat(x.sum(0)) - (0 if y is None else stat(y.sum(0)))
    r = {"value": r4(val), "lo": r4(np.nanquantile(boots, 0.025)), "hi": r4(np.nanquantile(boots, 0.975)), "n": n}
    if y is not None:
        r["p_a_not_better"] = r4((boots <= 0).mean())
    return r


def stage_leaf_types(cfg, run_dir):
    leaf = np.load(cfg["leaf_probs"])
    lf = pd.DataFrame({"screen_id": leaf["screen_id"], "node_idx": leaf["node_idx"], "label": leaf["label"]})
    order, ties = type_row_order()
    src = {k: np.load(v) for k, v in cfg["type_sources"].items()}
    checks = {"type_rows": len(order), "tied_positions": ties}
    for k, f in src.items():                                    # same rows, same labels, same screens
        assert (f["screen_id"] == order.screen_id.to_numpy()).all(), k
        assert (f["y"] == order.label.to_numpy()).all(), k
    pos = order.reset_index().merge(lf.reset_index(), on=["screen_id", "node_idx"], how="right",
                                    validate="one_to_one", suffixes=("_t", "_l"))
    assert pos["index_t"].notna().all() and (pos.label_t == pos.label_l).all()
    assert (pos.index_l.to_numpy() == np.arange(len(lf))).all()
    rows = pos.index_t.to_numpy(np.int64)
    # rows sharing a position could only be swapped among themselves; count leaves where that could matter
    boxes = pd.read_parquet("data/processed/nodes.parquet", columns=["screen_id", "node_idx", "nx0", "ny0", "nx1", "ny1"],
                            filters=[("screen_id", "in", sorted(order.screen_id.unique().tolist()))])
    el_pos = order.merge(boxes, on=["screen_id", "node_idx"], how="left", validate="one_to_one")
    dup = el_pos.duplicated(["screen_id", "nx0", "ny0", "nx1", "ny1"], keep=False).to_numpy()
    tied_leaves = int(dup[rows].sum())
    del boxes, el_pos
    checks.update({"leaves": len(lf), "leaf_screens": int(lf.screen_id.nunique()),
                   "all_leaves_matched_with_equal_labels": True,
                   "leaves_sharing_an_identical_box_with_another_element": tied_leaves})
    y, sid = lf.label.to_numpy(np.int64), lf.screen_id.to_numpy()
    support = np.bincount(y, minlength=24)

    def pred(source, key, valid_known=False):
        """valid_known: the invalid class (0) cannot be predicted, i.e. the model is told the element is
        valid. Models 4-6 never saw invalid elements in training (leaves are valid), Models 1-3 did."""
        if source == "leaf":
            p = leaf[key]
        else:
            p = src[source][key][rows]
        if p.ndim == 1:                                         # labels only (lookup): left as is
            return p.astype(np.int64)
        p = p.astype(np.float32)
        if valid_known:
            p[:, 0] = -np.inf
        return p.argmax(1).astype(np.int64)

    def full_macro(source, key):                                # all test elements, to check against Models 1-3
        f = src[source]
        p = f[key].astype(np.float32).argmax(1) if f[key].ndim == 2 else f[key]
        return type_scores(per_screen_confusion(f["screen_id"], f["y"].astype(np.int64),
                                                p.astype(np.int64)).sum(0))["macro_f1_types"]

    small = [t for t in range(1, 24) if support[t] < cfg["small_class_min_support"]]
    keep = [t for t in range(1, 24) if t not in small]

    def macro_keep(cm):
        full = np.zeros((25, 25), np.int64)
        full[:24, :24] = cm
        tp = np.diag(full)[:24].astype(float)
        fp, fn = full[:, :24].sum(0) - tp, full[:24, :].sum(1) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        return float(np.nanmean(f1[keep]))

    def macro(cm):
        return type_scores(cm)["macro_f1_types"]

    def score(p):
        cm = per_screen_confusion(sid, y, p)
        tot = cm.sum(0)
        tp = np.diag(tot)
        per = {TYPE_NAMES[t]: r4(2 * tp[t] / (tot[t].sum() + tot[:, t].sum())) for t in range(1, 24)}
        return cm, {"macro_f1_types": fast_boot(cm, macro, cfg["n_boot"], cfg["seed"]),
                    f"macro_f1_types_support_ge_{cfg['small_class_min_support']}": r4(macro_keep(tot)),
                    "accuracy": r4(tp.sum() / tot.sum()), "predicted_invalid": int(tot[:, 0].sum()),
                    "per_class_f1": per}

    out = {}
    for mode in ["as_trained", "valid_known"]:
        vk = mode == "valid_known"
        cms, res = {}, {}
        for name, (source, key) in cfg["systems"].items():
            cms[name], res[name] = score(pred(source, key, vk))
            if source != "leaf" and not vk:
                res[name]["macro_f1_types_all_elements"] = r4(full_macro(source, key))
        seeds = {}
        for name, members in cfg["seed_sets"].items():
            v = [macro(per_screen_confusion(sid, y, pred(s, k, vk)).sum(0)) for s, k in members]
            seeds[name] = {"values": [r4(x) for x in v], "mean": r4(np.mean(v)), "std": r4(np.std(v))}
        paired = {f"{a}_minus_{b}": fast_boot(cms[a], macro, cfg["n_boot"], cfg["seed"], y=cms[b])
                  for a, b in cfg["leaf_pairs"]}
        out[mode] = {"systems": res, "seeds": seeds, "paired": paired}
    return {"checks": checks, "support": {TYPE_NAMES[t]: int(support[t]) for t in range(1, 24)},
            "small_types_excluded_in_sensitivity": [TYPE_NAMES[t] for t in small], **out,
            "notes": ["Leaves = valid elements with no valid child; invalid F1 cannot be measured on them.",
                      "Models 1-3 were trained on all elements; Models 4-6 type heads on leaves only "
                      "(auxiliary loss, weight 0.2).",
                      "as_trained: argmax over all 24 outputs, so Models 1-3 can call a valid leaf invalid. "
                      "valid_known: the invalid output is masked for every model (Models 4-6 unchanged: "
                      "they never predict it); like-for-like. The lookup baseline gives labels only and "
                      "is not masked."]}


def print_leaf(m):
    print("checks", m["checks"])
    print("small types (sensitivity excludes):", m["small_types_excluded_in_sensitivity"])
    for mode in ["as_trained", "valid_known"]:
        sm = m[mode]["systems"]
        k_small = [k for k in next(iter(sm.values())) if k.startswith("macro_f1_types_support")][0]
        print(f"\n--- {mode}")
        print(f"{'system':20s} {'leaf macro':>10s} {'[95% CI]':>17s} {'w/o small':>9s} {'acc':>6s} {'pred inv':>8s} "
              f"{'all elems':>9s}")
        for n, r in sm.items():
            mf = r["macro_f1_types"]
            al = r.get("macro_f1_types_all_elements")
            print(f"{n:20s} {mf['value']:10.4f} [{mf['lo']:.4f}, {mf['hi']:.4f}] {r[k_small]:9.4f} {r['accuracy']:6.4f} "
                  f"{r['predicted_invalid']:8d} {'' if al is None else f'{al:9.4f}'}")
        for n, s in m[mode]["seeds"].items():
            print(f"seeds {n}: {s['values']} mean {s['mean']} std {s['std']}")
        for n, p in m[mode]["paired"].items():
            print(f"  {n:24s} {p['value']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] p={p['p_a_not_better']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rescore", "leaf_types"])
    ap.add_argument("--config", default="configs/analyse_saved_outputs.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    run_dir = start_run(f"saved_outputs_{a.stage}", {**cfg, "stage": a.stage})
    m = stage_rescore(cfg, run_dir) if a.stage == "rescore" else stage_leaf_types(cfg, run_dir)
    finish_run(run_dir, m)
    (print_rescore if a.stage == "rescore" else print_leaf)(m)
    print(f"\nrun folder: {run_dir}")


if __name__ == "__main__":
    main()
