"""Build data/processed/{nodes,screens}.parquet from RICO + CLAY and check every count.

Run from the project root:  python scripts/build_node_table.py --config configs/build_node_table.yaml
Writes the check report (config, environment, measured vs expected) to results/node_table_check.json.
"""
import argparse
import json
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image

from uistruct.data.rico_clay import load_screen

STR_COLS = ("pointer", "class_name", "resource_id", "text", "content_desc", "visibility")


def work(args):
    rico_dir, chunk = args
    dfs, metas = [], []
    for sid, labels in chunk:
        df, meta = load_screen(rico_dir, sid, labels)
        with Image.open(Path(rico_dir) / f"{sid}.jpg") as im:  # header only, no decode
            meta["img_w"], meta["img_h"] = im.size
        dfs.append(df)
        metas.append(meta)
    df = pd.concat(dfs, ignore_index=True)
    # fixed string type, so a chunk where a text column is all null still matches the file schema
    schema = pa.Schema.from_pandas(df, preserve_index=False)
    for c in STR_COLS:
        schema = schema.set(schema.get_field_index(c), pa.field(c, pa.string()))
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False), metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/build_node_table.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    t0 = time.time()
    clay, out = Path(cfg["clay_dir"]), Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # node_id must be read as text: ids are 3-7 hex chars and some are all digits
    lab = pd.read_csv(clay / "clay_labels.csv", dtype={"node_id": str})
    n_exact_dup = int(lab.duplicated().sum())
    n_conflict = int(lab.drop_duplicates().duplicated(["screen_id", "node_id"]).sum())
    assert n_conflict == 0, "same CLAY node with two different labels"
    lab = lab.drop_duplicates()
    names = {
        int(k): v.strip() for k, v in (l.split(":") for l in (clay / "label_map.txt").read_text().splitlines() if l)}
    split_of = {}
    for s in ("train", "dev", "test"):
        for i in (clay / f"split_{s}_id.txt").read_text().split():
            split_of[int(i)] = s

    groups = {sid: dict(zip(g.node_id, g.label)) for sid, g in lab.groupby("screen_id")}
    sids = sorted(groups)
    items = [(s, groups[s]) for s in sids][: cfg.get("limit")]
    chunks = [(cfg["rico_dir"], items[i:i + cfg["chunk"]]) for i in range(0, len(items), cfg["chunk"])]

    writer, metas = None, []
    with Pool(cfg["workers"]) as pool:
        for k, (table, m) in enumerate(pool.imap(work, chunks)):  # imap keeps screen order
            if writer is None:
                writer = pq.ParquetWriter(out / "nodes.parquet", table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
            metas += m
            if k % 20 == 0:
                print(f"{k + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)
    writer.close()

    screens = pd.DataFrame(metas)
    screens["split"] = screens.screen_id.map(split_of)
    # 35 screenshots are landscape while their hierarchy is mostly in the portrait frame: boxes do not
    # line up with pixels. Flagged, not dropped (each image model decides whether to skip them).
    screens["img_landscape"] = screens.img_w > screens.img_h
    screens.to_parquet(out / "screens.parquet", index=False)

    # ---- checks (read back what was written, so the check covers the files themselves) ----
    cols = ["screen_id", "node_idx", "parent_idx", "clay_parent_idx", "label", "is_labelled",
            "visible_to_user", "box_ok", "dup_pointer", "depth", "nx0", "ny0", "nx1", "ny1"]
    n = pd.read_parquet(out / "nodes.parquet", columns=cols)
    L = n[n.is_labelled]
    per = L.groupby("screen_id").size()
    exp = cfg["expected"]
    ratio = screens.img_w / screens.img_h
    nonroot = L[L.label != -1]
    parent_lab = n.is_labelled.values[  # is the raw parent of each labelled non-root node labelled?
        (nonroot.index.values - nonroot.node_idx.values + nonroot.parent_idx.values)]
    got = {
        "clay_rows_raw": int(len(lab) + n_exact_dup),
        "clay_exact_duplicate_rows_removed": n_exact_dup,
        "clay_unique_labelled_nodes": int(len(lab)),
        "screens": int(len(screens)),
        "screens_by_split": screens.split.value_counts().to_dict(),
        "screens_without_split": int(screens.split.isna().sum()),
        "rico_nodes_total": int(len(n)),
        "labelled_rows_in_table": int(len(L)),
        "labels_unmatched": int(screens.n_labels_unmatched.sum()),
        "dup_pointer_rows": int(n.dup_pointer.sum()),
        "screens_with_dup_pointer": int((screens.n_dup_pointer_rows > 0).sum()),
        "dup_pointer_rows_whose_pointer_is_labelled": None,  # filled below
        "labelled_nodes_by_split": L.screen_id.map(split_of).value_counts().to_dict(),
        "labelled_per_screen": {"mean": round(float(per.mean()), 2), "median": float(per.median()),
                                "p90": float(per.quantile(0.9)), "max": int(per.max())},
        "rico_nodes_per_screen": {"mean": round(float(screens.n_nodes.mean()), 2),
                                  "median": float(screens.n_nodes.median()), "max": int(screens.n_nodes.max())},
        "screens_with_background": int(L[L.label == 0].screen_id.nunique()),
        "background_share_of_labelled": round(float((L.label == 0).mean()), 4),
        "class_share_pct": {names[k]: round(100 * v, 2) for k, v in L.label.value_counts(normalize=True).items()},
        "labelled_not_visible_to_user": int((~L.visible_to_user).sum()),
        "labelled_box_not_ok": int((~L.box_ok).sum()),
        "labelled_nonroot_with_unlabelled_raw_parent_pct": round(100 * float((~parent_lab).mean()), 2),
        "labelled_nonroot_without_labelled_ancestor": int((nonroot.clay_parent_idx < 0).sum()),
        "roots_labelled_ROOT": int(((n.node_idx == 0) & (n.label == -1)).sum()),
        "ROOT_label_not_at_node0": int(((n.node_idx != 0) & (n.label == -1)).sum()),
        "depth_max": int(n.depth.max()),
        "image_sizes": {f"{w}x{h}": int(c) for (w, h), c in screens.groupby(["img_w", "img_h"]).size().items()},
        "images_not_9x16": int((np.abs(ratio - 9 / 16) > 1e-3).sum()),
        "labelled_box_outside_unit_frame": int(((L.nx0 < 0) | (L.ny0 < 0) | (L.nx1 > 1) | (L.ny1 > 1)).sum()),
    }
    dp = pd.read_parquet(out / "nodes.parquet", columns=["screen_id", "pointer", "dup_pointer"])
    dups = dp[dp.dup_pointer]
    labset = set(zip(lab.screen_id, lab.node_id))
    got["dup_pointer_rows_whose_pointer_is_labelled"] = int(
        sum((s, p) in labset for s, p in zip(dups.screen_id, dups.pointer)))
    # proposal-era join, recounted the same way: every raw CLAY row x every RICO copy of its pointer
    raw = pd.read_csv(clay / "clay_labels.csv", dtype={"node_id": str})
    copies = dp.groupby(["screen_id", "pointer"]).size().rename("k").reset_index()
    got["join_rows_raw_clay_x_rico_copies"] = int(raw.merge(
        copies, left_on=["screen_id", "node_id"], right_on=["screen_id", "pointer"]).k.sum())
    got["screens_img_landscape"] = int(screens.img_landscape.sum())
    got["screens_without_ROOT_label"] = int(len(screens) - got["roots_labelled_ROOT"])

    checks = {
        "clay_rows": got["clay_rows_raw"] == exp["clay_rows"],
        "screens": got["screens"] == exp["clay_screens"],
        "splits": got["screens_by_split"] == {"train": exp["split_train"], "dev": exp["split_dev"],
                                              "test": exp["split_test"]},
        "every_label_matched": got["labels_unmatched"] == 0,
        "labelled_rows_equal_unique_clay_nodes": got["labelled_rows_in_table"] == got["clay_unique_labelled_nodes"],
        "join_with_duplicates_reproduced": got["join_rows_raw_clay_x_rico_copies"] == exp["join_rows_with_duplicates"],
        "screens_with_background": got["screens_with_background"] == exp["screens_with_background"],
        "labelled_per_screen": (round(got["labelled_per_screen"]["mean"], 1) == exp["labelled_per_screen_mean"]
                                and got["labelled_per_screen"]["median"] == exp["labelled_per_screen_median"]
                                and got["labelled_per_screen"]["p90"] == exp["labelled_per_screen_p90"]
                                and got["labelled_per_screen"]["max"] == exp["labelled_per_screen_max"]),
        "images_9x16_except_flagged_landscape": int((np.abs(ratio - 9 / 16) > 1e-3).sum())
        == got["screens_img_landscape"] and bool((np.abs(ratio[screens.img_landscape] - 16 / 9) < 1e-3).all()),
    }
    report = {"config": cfg, "seconds": round(time.time() - t0, 1),
              "env": {"python": sys.version.split()[0], "pandas": pd.__version__, "pyarrow": pa.__version__,
                      "platform": platform.platform()},
              "checks": checks, "all_checks_pass": all(checks.values()), "measured": got}
    Path(cfg["results"]).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(checks, indent=2), "\nall pass:", all(checks.values()), f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
