"""The tables the nesting stages need on Colab, as one small tar for Google Drive.

  python scripts/pack_colab_data_structure.py   ->  data/upload/structure_colab_tables.tar (+ manifest with sha256)

Contents (clean-split screens only; rows and values identical to data/processed, so a Colab build gives the
same caches as a laptop build, checked by the build stage's fingerprint):
  processed/elements.parquet  clean-split elements: screen_id, node_idx, gt_parent, label, is_valid, split_clean, box
  processed/nodes.parquet     every raw RICO node of those screens: the columns of train_structure_models.RAW_COLS
Screenshots are not in it: Colab downloads the RICO archive itself (as for Model 3, checked byte for byte).
"""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train_structure_models import BOX, RAW_COLS  # noqa: E402

PROC = Path("data/processed")
OUT = Path("data/upload/structure_colab_tables.tar")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


if __name__ == "__main__":
    slim = OUT.parent / "slim_structure"
    slim.mkdir(parents=True, exist_ok=True)
    el = pd.read_parquet(PROC / "elements.parquet",
                         columns=["screen_id", "node_idx", "gt_parent", "label", "is_valid", "split_clean", *BOX])
    el = el[el.split_clean.isin(["train", "dev", "test"])]
    sids = set(el.screen_id.unique())
    nodes = pd.read_parquet(PROC / "nodes.parquet", columns=RAW_COLS)
    nodes = nodes[nodes.screen_id.isin(sids)]
    el.to_parquet(slim / "elements.parquet", index=False, compression="zstd")
    nodes.to_parquet(slim / "nodes.parquet", index=False, compression="zstd")
    with tarfile.open(OUT, "w") as tar:
        for f in ["elements.parquet", "nodes.parquet"]:
            tar.add(slim / f, arcname=f"processed/{f}")
    man = {"file": OUT.name, "bytes": OUT.stat().st_size, "sha256": sha(OUT), "n_screens": len(sids),
           "n_elements": len(el), "n_nodes": len(nodes)}
    (OUT.parent / "structure_colab_tables_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
