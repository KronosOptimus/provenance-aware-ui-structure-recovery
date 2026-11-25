"""Check the raw CLAY and RICO downloads and write the counts to results/raw_data_check.json.

Run from the project root:  python scripts/check_raw_data.py [--md5]
Expected values come from an earlier exploratory run; any mismatch is reported.
"""
import argparse
import base64
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

RAW = Path("data/raw")
OUT = Path("results/raw_data_check.json")

EXPECTED = {
    "clay_rows": 1_427_915,
    "clay_screens": 59_555,
    # Note: the split files have no trailing newline, so the earlier
    # line counts (44,628 / 6,206 / 8,718) missed one id each. Ids sum to 59,555.
    "clay_split_train": 44_629,
    "clay_split_dev": 6_207,
    "clay_split_test": 8_719,
    "clay_label_ids": 25,  # 24 types + ROOT
    "rico_json": 66_261,
    "rico_tar_bytes": 6_471_262_799,
    "rico_tar_md5_b64": "vc+Z1Nx/NqM5blfjBsdWlg==",
}


def md5_b64(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return base64.b64encode(h.digest()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md5", action="store_true", help="hash the 6.5 GB RICO archive (slow)")
    args = ap.parse_args()

    got = {}

    # CLAY
    clay = RAW / "clay"
    labels = pd.read_csv(clay / "clay_labels.csv")
    got["clay_rows"] = len(labels)
    got["clay_screens"] = int(labels.screen_id.nunique())
    splits = {}
    for s in ("train", "dev", "test"):
        ids = [int(x) for x in (clay / f"split_{s}_id.txt").read_text().split()]
        splits[s] = set(ids)
        got[f"clay_split_{s}"] = len(ids)
    got["clay_splits_disjoint"] = not (splits["train"] & splits["dev"] or splits["train"] & splits["test"]
                                       or splits["dev"] & splits["test"])
    got["clay_label_ids"] = len((clay / "label_map.txt").read_text().strip().splitlines())
    clay_screens = set(labels.screen_id.unique())
    got["clay_screens_not_in_any_split"] = len(clay_screens - set().union(*splits.values()))

    # RICO
    tar = RAW / "rico" / "unique_uis.tar.gz"
    got["rico_tar_bytes"] = tar.stat().st_size if tar.exists() else None
    if args.md5 and tar.exists():
        got["rico_tar_md5_b64"] = md5_b64(tar)
    combined = RAW / "rico" / "combined"
    names = os.listdir(combined)
    json_ids = {int(n[:-5]) for n in names if n.endswith(".json")}
    jpg_ids = {int(n[:-4]) for n in names if n.endswith(".jpg")}
    got["rico_json"] = len(json_ids)
    got["rico_jpg"] = len(jpg_ids)
    got["rico_json_without_jpg"] = len(json_ids - jpg_ids)
    got["rico_jpg_without_json"] = len(jpg_ids - json_ids)
    got["clay_screens_missing_json"] = len(clay_screens - json_ids)
    got["clay_screens_missing_jpg"] = len(clay_screens - jpg_ids)
    got["rico_extracted_bytes"] = sum((combined / n).stat().st_size for n in names)

    # Screenshot sizes on a fixed sample (first 200 CLAY train ids, sorted)
    sample = sorted(splits["train"] & jpg_ids)[:200]
    sizes = {}
    for i in sample:
        with Image.open(combined / f"{i}.jpg") as im:
            sizes[f"{im.size[0]}x{im.size[1]}"] = sizes.get(f"{im.size[0]}x{im.size[1]}", 0) + 1
    got["screenshot_sizes_sample200"] = sizes

    mismatches = {k: {"expected": v, "got": got.get(k)} for k, v in EXPECTED.items()
                  if k in got and got[k] != v}
    report = {
        "task": "raw_data_check",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "clay_commit": subprocess.run(["git", "-C", str(clay), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip() or None,
        "results": got,
        "expected": EXPECTED,
        "mismatches": mismatches,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
