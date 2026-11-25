"""App-disjoint splits and ground-truth elements/tree.

Run:  python scripts/build_splits_and_targets.py --config configs/build_splits_and_targets.yaml
Writes data/processed/splits.parquet, data/processed/elements.parquet and results/splits_and_targets.json.
Leakage checks (both must be 0 after cleaning): companies shared between splits, and train screens
with a layout identical to a dev/test screen (same labels + boxes rounded to 1% of the frame).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uistruct.data.splits import clean_splits, overlap_report
from uistruct.data.tree import NO_PARENT, VROOT, build_elements


def layout_hash(df):
    """Hash of a screen's labelled layout: boxes rounded to 1% of the frame, plus labels."""
    r = df[["label", "nx0", "ny0", "nx1", "ny1"]].copy()
    r[["nx0", "ny0", "nx1", "ny1"]] = (r[["nx0", "ny0", "nx1", "ny1"]] * 100).round()
    r = r.sort_values(list(r.columns))
    return hashlib.md5(r.to_numpy().astype(np.int64).tobytes()).hexdigest()


def main(cfg):
    t0 = time.time()
    d = Path(cfg["data_dir"])
    screens = pd.read_parquet(d / "screens.parquet")
    nodes = pd.read_parquet(d / "nodes.parquet", columns=["screen_id", "node_idx", "clay_parent_idx", "label",
                                                          "is_labelled", "nx0", "ny0", "nx1", "ny1"])
    el = build_elements(nodes)
    lh = el.groupby("screen_id").apply(layout_hash, include_groups=False)
    s = clean_splits(screens, lh, cfg["drop_dev_test_overlap"])
    s["layout_hash"] = s.screen_id.map(lh)
    rep = {"official_overlap": {u: overlap_report(s, "split", u) for u in ("app", "company")},
           "clean_overlap": {u: overlap_report(s, "split_clean", u) for u in ("app", "company")}}
    rep["screens"] = {"official": s.split.value_counts().to_dict(), "clean": s.split_clean.value_counts().to_dict()}
    assert all(v["shared_company"] == 0 for v in rep["clean_overlap"]["company"].values()), "company leakage remains"
    ev = s.split_clean.isin(["dev", "test"])
    rep["layout_duplicates_train_vs_eval_after_cleaning"] = int(
        s[s.split_clean == "train"].layout_hash.isin(set(s[ev].layout_hash)).sum())
    assert rep["layout_duplicates_train_vs_eval_after_cleaning"] == 0
    rep["layout_duplicates_train_vs_eval_official"] = int(
        s[s.split == "train"].layout_hash.isin(set(s[s.split.isin(["dev", "test"])].layout_hash)).sum())

    el = el.merge(s[["screen_id", "split_clean"]], on="screen_id")
    valid = el[el.is_valid]
    rep["elements"] = {
        "elements": len(el), "valid": len(valid), "invalid": int((~el.is_valid).sum()),
        "valid_parent_vroot": int((valid.gt_parent == VROOT).sum()),
        "invalid_have_no_parent": bool((el[~el.is_valid].gt_parent == NO_PARENT).all()),
        "per_split_elements": el.split_clean.value_counts().to_dict(),
        "per_split_valid": valid.split_clean.value_counts().to_dict(),
        "train_label_counts": el[el.split_clean == "train"].label.value_counts().sort_index().to_dict(),
    }
    # reattached = true parent differs from the raw CLAY parent (an invalid node was skipped)
    raw_parent = valid.clay_parent_idx
    root_idx = nodes[nodes.label == -1].set_index("screen_id").node_idx
    to_root = raw_parent.to_numpy() == valid.screen_id.map(root_idx).fillna(-99).to_numpy()
    moved = ~((valid.gt_parent == raw_parent) | ((valid.gt_parent == VROOT) & (to_root | (raw_parent < 0))))
    rep["elements"]["reattached_valid"] = int(moved.sum())
    rep["elements"]["reattached_screens"] = int(valid[moved.to_numpy()].screen_id.nunique())
    nv = valid.groupby("screen_id").size()
    rep["elements"]["valid_per_screen"] = {"mean": round(float(nv.mean()), 2), "median": float(nv.median()),
                                           "max": int(nv.max())}

    s.to_parquet(d / "splits.parquet", index=False)
    el.to_parquet(d / "elements.parquet", index=False)
    rep["config"] = cfg
    rep["seconds"] = round(time.time() - t0, 1)
    Path(cfg["results"]).write_text(json.dumps(rep, indent=2, default=int))
    print(json.dumps(rep, indent=2, default=int))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(yaml.safe_load(open(ap.parse_args().config)))
