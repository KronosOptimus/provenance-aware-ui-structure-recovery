"""Extract the colour features (src/uistruct/data/colour.py) for every element of the clean splits,
once, cached to data/processed/colour_features.parquet. Summary counts go to the run folder.

Run: python scripts/extract_colour_features.py --config configs/colour_features.yaml
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

from uistruct.data.colour import COLOUR, screen_features
from uistruct.runlog import finish_run, start_run


def job(args):
    path, sid, idx, boxes, strip, ring = args
    return sid, idx, screen_features(path, boxes, strip, ring)


def main(cfg, run_dir):
    el = pd.read_parquet(f"{cfg['data_dir']}/elements.parquet", columns=["screen_id", "node_idx", "label", "split_clean"])
    el = el[el.split_clean.isin(cfg["splits"])]
    nodes = pd.read_parquet(f"{cfg['data_dir']}/nodes.parquet", columns=["screen_id", "node_idx", "nx0", "ny0", "nx1", "ny1"])
    el = el.merge(nodes, on=["screen_id", "node_idx"], how="left", validate="one_to_one")
    jobs = [(f"{cfg['image_dir']}/{sid}.jpg", sid, g.node_idx.to_numpy(), g[["nx0", "ny0", "nx1", "ny1"]].to_numpy(np.float64),
             cfg["strip_px"], cfg["ring_px"]) for sid, g in el.groupby("screen_id")]
    t0, parts = time.time(), []
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        for i, (sid, idx, f) in enumerate(ex.map(job, jobs, chunksize=64)):
            parts.append(pd.DataFrame(f, columns=COLOUR).assign(screen_id=sid, node_idx=idx))
            if i % 5000 == 0:
                print(f"{i}/{len(jobs)} screens ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)[["screen_id", "node_idx", *COLOUR]]
    out.to_parquet(f"{cfg['data_dir']}/colour_features.parquet", index=False)
    m = out.merge(el[["screen_id", "node_idx", "label", "split_clean"]], on=["screen_id", "node_idx"])
    assert len(m) == len(el) and not out[COLOUR].isna().any().any()
    return {"n_screens": len(jobs), "n_elements": len(out), "seconds": round(time.time() - t0),
            "no_image_by_split": m.groupby("split_clean").no_image.agg(["sum", "size"]).to_dict("index"),
            "no_image_by_validity": {"invalid": float(m[m.label == 0].no_image.mean()), "valid": float(m[m.label > 0].no_image.mean())},
            "border_missing_share": float(m.border_missing.mean()),
            "feature_means_train": m[m.split_clean == "train"][COLOUR].mean().round(3).to_dict()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    run_dir = start_run("colour_features", cfg)
    res = main(cfg, run_dir)
    finish_run(run_dir, res)
    print(res, "\nrun dir:", run_dir)
