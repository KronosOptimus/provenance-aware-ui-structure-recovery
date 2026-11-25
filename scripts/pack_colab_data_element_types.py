"""Pack the data Colab needs into one file for Google Drive (run on the laptop).

Contents: the original screenshots of the 47,088 clean-split screens (screens/<id>.jpg, unchanged bytes,
so pixels are identical to the colour features) and the processed tables (processed/*.parquet). JPEGs are stored, not
recompressed. Writes data/upload/element_types_colab_data.tar and a manifest (file count, bytes, sha256) next to it;
the Colab notebook checks the sha256 after copying.

  python scripts/pack_colab_data_element_types.py           # screenshots + tables (NOT used: see --tables)
  python scripts/pack_colab_data_element_types.py --extra   # small file: Model 2 dev/test probabilities + CLAY label map
                                                  # (paired comparisons with Model 2 on the same rows)
  python scripts/pack_colab_data_element_types.py --tables  # small file: slim tables (clean-split elements only) + the
                                                  # sha256 of the screenshots Colab must end up with

Note: uploading the 5.7 GB file would take ~30 h, so Colab downloads the
original RICO archive itself (Google Cloud Storage, md5 checked) and keeps only the clean-split screenshots;
their bytes are checked against `screens_sha256` computed here from the laptop copy.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/rico/combined")
PROC = Path("data/processed")
TABLES = ["elements.parquet", "splits.parquet", "nodes.parquet", "screens.parquet", "colour_features.parquet"]
OUT = Path("data/upload/element_types_colab_data.tar")


EXTRA = {"element_types_colour_dev_probs.npz": "results/runs/element_types_colour_dev_run1/dev_probs.npz",
         "element_types_colour_test_probs.npz": "results/runs/element_types_colour_test_run1/test_probs.npz",
         "label_map.txt": "data/raw/clay/label_map.txt"}


def extra():
    out = OUT.parent / "element_types_colab_extra.tar"
    with tarfile.open(out, "w") as tar:
        for name, src in EXTRA.items():
            tar.add(src, arcname=f"extra/{name}")
    print(out, out.stat().st_size)


def screens_sha256(ids):
    """One sha256 over (id, file bytes) of every screenshot, in sorted id order."""
    h = hashlib.sha256()
    for i in ids:
        h.update(f"{i}:".encode())
        h.update((RAW / f"{i}.jpg").read_bytes())
    return h.hexdigest()


def tables():
    from uistruct.data.features import FLAGS
    slim = OUT.parent / "slim"
    slim.mkdir(parents=True, exist_ok=True)
    el = pd.read_parquet(PROC / "elements.parquet", columns=["screen_id", "node_idx", "label", "is_valid", "split_clean"])
    el = el[el.split_clean.isin(["train", "dev", "test"])]
    nodes = pd.read_parquet(PROC / "nodes.parquet", columns=["screen_id", "node_idx", "depth", "n_children", "class_name",
                                                             "nx0", "ny0", "nx1", "ny1", *FLAGS])
    nodes = nodes.merge(el[["screen_id", "node_idx"]], on=["screen_id", "node_idx"], validate="one_to_one")
    el.to_parquet(slim / "elements.parquet", index=False)
    nodes.to_parquet(slim / "nodes.parquet", index=False)
    pd.read_parquet(PROC / "colour_features.parquet").to_parquet(slim / "colour_features.parquet", index=False)
    ids = sorted(el.screen_id.unique())
    (slim / "screen_ids.txt").write_text("\n".join(map(str, ids)))
    out = OUT.parent / "element_types_colab_tables.tar"
    with tarfile.open(out, "w") as tar:
        for f in ["elements.parquet", "nodes.parquet", "colour_features.parquet", "screen_ids.txt"]:
            tar.add(slim / f, arcname=f"processed/{f}")
    man = {"file": out.name, "tar_bytes": out.stat().st_size, "n_screens": len(ids), "n_elements": len(el),
           "screens_sha256": screens_sha256(ids)}
    (OUT.parent / "element_types_colab_tables_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(PROC / "splits.parquet")
    ids = sorted(s.loc[s.split_clean != "dropped", "screen_id"].unique())
    n_bytes = 0
    with tarfile.open(OUT, "w") as tar:
        for t in TABLES:
            tar.add(PROC / t, arcname=f"processed/{t}")
            n_bytes += (PROC / t).stat().st_size
        for k, i in enumerate(ids):
            p = RAW / f"{i}.jpg"
            tar.add(p, arcname=f"screens/{i}.jpg")
            n_bytes += p.stat().st_size
            if k % 5000 == 0:
                print(k, "/", len(ids), flush=True)
    h = hashlib.sha256()
    with open(OUT, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    man = {"file": OUT.name, "n_screens": len(ids), "tables": TABLES, "content_bytes": n_bytes,
           "tar_bytes": OUT.stat().st_size, "sha256": h.hexdigest()}
    (OUT.parent / "element_types_colab_data_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    extra() if a.extra else tables() if a.tables else main()
