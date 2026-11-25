"""Android-to-web test: the final Android-trained system (no retraining, no re-tuning) on WebUI pages.
Config: configs/evaluate_android_system_on_web_pages.yaml.

Stages
  fetch               pick pages (seeded) from WebUI's official validation and test splits; download only their files
  build --split s     each page and device -> the provenance inputs at the levels in the config (data/webui.py)
  run   --split s     crop embeddings (Model 3's backbone), the provenance network, rule baselines, scores -> a run folder.
                      --split test needs --confirm-test and runs once.
System and thresholds are the planned provenance system as the demo runs it (fusion_full_s0, t_c and t_e from the Android
validation choices stored in the Android test run). Nothing is fitted or chosen on web pages.
"""
import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from uistruct.data import provenance_tokens as PT
from uistruct.data import webui as W
from uistruct.data.features import FLAGS
from uistruct.data.nesting_crops import SIZE, screen_boxes
from uistruct.data.nesting_tokens import rule_trees
from uistruct.eval.metrics import VROOT, density_bin, grouping_metrics, smooth_single_child
from uistruct.eval.stats import bootstrap_mean, paired_bootstrap
from uistruct.models.cnn import Normalise, resnet18_trunk
from uistruct.models.nesting_raw import prune, raw_chain
from uistruct.runlog import finish_run, start_run
from uistruct.train import provenance_head as PH

CROP_DIM = 512


def flag_fill(cfg):
    return {f: bool(cfg["flag_fill"].get(f, False)) for f in FLAGS}


# ---------------------------------------------------------------- fetch
def stage_fetch(cfg):
    out = {}
    for split, n in [("val", cfg["n_val_pages"]), ("test", cfg["n_test_pages"])]:
        rz = W.RemoteZip(split)
        pages = W.page_ids(rz)
        ok = W.eligible(pages)
        pick = sorted(np.random.default_rng(cfg["seed"]).choice(ok, n, replace=False).tolist())
        print(f"{split}: {len(pages)} pages in archive, {len(ok)} with both devices complete; fetching {n}", flush=True)
        dst = Path(cfg["raw_dir"]) / split
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "pages.json").write_text(json.dumps(pick))
        out[split] = {"archive_pages": len(pages), "eligible": len(ok), "picked": n,
                      **W.fetch_pages(rz, pick, dst, cfg["fetch_threads"])}
    print(json.dumps(out), flush=True)
    Path(cfg["cache_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["cache_dir"]) / "fetch_summary.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- build
def _build_one(a):
    page_dir, device, page, cfg = a
    g, raw, info = W.page_tables(page_dir, device, int(page), flag_fill(cfg))
    if g is None:
        return page, device, None, info
    fusion = yaml.safe_load(open(cfg["fusion_config"]))
    seed = cfg["corruption_seed"]
    sc = PT.build_screen_levels(g, raw, fusion["pool"], fusion["min_iou"], cfg["levels"], {"p_inj": 0.0}, None, seed)
    true = dict(zip(g.node_idx.tolist(), g.gt_parent.tolist()))
    leaves = set(true) - set(true.values())
    lf = g[g.node_idx.isin(leaves)]
    rules = {"xycut": rule_trees(g, None, cfg["xy_tol"])["xycut"], "flat": {l: VROOT for l in leaves}}
    for lv in cfg["levels"]:
        if lv == "absent":
            continue
        s = PT.level_settings(lv, {"p_inj": 0.0})
        rng = np.random.default_rng([seed, int(page), PT.LEVEL_ID[lv]])     # as build_screen_levels draws it
        r = PT.drop_raw(raw, set(leaves), s["p_drop"], rng)
        rules[f"raw_collapsed|{lv}"] = smooth_single_child(prune(raw_chain(r, leaves), leaves), leaves)
    smooth_same = (sorted(map(sorted, (s for _, s in _leafsets(true, leaves))))
                   == sorted(map(sorted, (s for _, s in _leafsets(smooth_single_child(true), leaves)))))
    info.update(n_leaves=len(lf), smoothing_changes_truth=not smooth_same)
    return page, device, {"sc": sc, "rules": rules}, info


def _leafsets(par, leaves):
    from uistruct.models.nesting import leaf_sets_of
    return leaf_sets_of(par, leaves)


def cache_file(cfg, split):
    return Path(cfg["cache_dir"]) / f"web_{split}.pkl"


def stage_build(cfg, split):
    t0 = time.time()
    pages = json.load(open(Path(cfg["raw_dir"]) / split / "pages.json"))
    jobs = [(str(Path(cfg["raw_dir"]) / split / p), d, p, cfg) for p in pages for d in cfg["devices"]]
    out, infos = {}, []
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        for page, device, item, info in ex.map(_build_one, jobs, chunksize=8):
            infos.append({"page": page, "device": device, **info})
            if item is not None:
                out[(page, device)] = item
    path = cache_file(cfg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"items": out, "infos": infos}, f, protocol=pickle.HIGHEST_PROTOCOL)
    inf = pd.DataFrame(infos)
    print(inf.groupby("device").agg(pages=("page", "size"), kept=("n_leaves", "count"),
                                    leaves_median=("n_leaves", "median")).to_string())
    print("smoothing changes truth on", int(inf.smoothing_changes_truth.fillna(False).sum()), "page-devices")
    print(f"build {split}: {len(out)} page-devices in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- run
class System:
    """The planned provenance system as the demo loads it (src/uistruct/demo.py)."""

    def __init__(self, cfg):
        st = torch.load(cfg["model"], map_location="cpu", weights_only=False)
        assert st["variant"] == "fusion_full" and st["seed"] == 0 and st["trust"], "not the planned provenance model"
        self.hp = st["hp"]
        self.enc = pickle.load(open(cfg["encoder"], "rb"))
        self.model = PH.build_fusion(self.hp, self.enc, CROP_DIM, True)
        self.model.load_state_dict(st["state"])
        self.model.eval()
        ck = torch.load(cfg["model3_ckpt"], map_location="cpu", weights_only=False)
        assert ck["fine_tune"], "not Model 3"
        self.trunk = resnet18_trunk(pretrained=False)
        self.trunk.load_state_dict({k[len("backbone."):]: v for k, v in ck["state"].items() if k.startswith("backbone.")})
        self.trunk.eval()
        self.norm = Normalise()
        tuned = json.load(open(cfg["android_test_run"]))["tuned"]
        self.t_c_plain = float(tuned["fusion_full_s0"]["t_c"])
        self.t_c, self.t_e = float(tuned["fusion_full_s0+trust"]["t_c"]), float(tuned["fusion_full_s0+trust"]["t_e"])


def _crop_one(a):
    path, device, boxes = a
    return W.web_crops(path, boxes, device, SIZE)


@torch.no_grad()
def embed(sysm, cfg, split, items, log=print):
    """Crop embeddings of every token box of every level, float32 on the CPU, stored float16 (as the demo)."""
    keys = sorted(items)
    jobs, idxs = [], []
    for k in keys:
        sc = items[k]["sc"]
        groups = {}
        for lv, v in sc["v"].items():
            groups[f"{lv}|leaf"] = v["leaf_xyxy"]
            groups[f"{lv}|M"] = v["M"]["crop_xyxy"]
        uniq, idx = screen_boxes(None, groups)
        jobs.append((str(Path(cfg["raw_dir"]) / split / k[0] / f"{k[1]}-screenshot.webp"), k[1], uniq))
        idxs.append(idx)
    out, t0 = {}, time.time()
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        for n, (k, idx, crops) in enumerate(zip(keys, idxs, ex.map(_crop_one, jobs, chunksize=4))):
            e = np.concatenate([sysm.trunk(sysm.norm(torch.from_numpy(crops[a:a + 512]))).numpy()
                                for a in range(0, len(crops), 512)]).astype(np.float16)
            out[k] = {name: e[ix] for name, ix in idx.items()}
            if (n + 1) % 200 == 0:
                log(f"  embedded {n + 1}/{len(keys)} page-devices ({time.time() - t0:.0f}s)")
    return out


def score_rows(cfg, sysm, items, emb):
    rows, trust_rows = [], []
    keys = sorted(items)
    for lv in cfg["levels"]:
        vs = [items[k]["sc"]["v"][lv] for k in keys]
        tr = [sysm.enc.transform(v, (emb[k][f"{lv}|leaf"], emb[k][f"{lv}|M"])) for k, v in zip(keys, vs)]
        cand, _, trust = PH.predict_fusion(sysm.model, tr, sysm.hp["budget"])
        for k, v, c, t in zip(keys, vs, cand, trust):
            L = len(v["leaf_ids"])
            has_meta = v["has_meta"]
            removed = (t[:L] < sysm.t_e) if has_meta else np.zeros(L, bool)
            preds = {"fusion_full_s0": PH.removal_tree(v, c, sysm.t_c_plain, np.zeros(L, bool)),
                     "fusion_full_s0+trust": PH.removal_tree(v, c, sysm.t_c, removed),
                     "xycut": items[k]["rules"]["xycut"], "flat": items[k]["rules"]["flat"]}
            if f"raw_collapsed|{lv}" in items[k]["rules"]:
                preds["raw_collapsed"] = items[k]["rules"][f"raw_collapsed|{lv}"]
            for name, par in preds.items():
                m = grouping_metrics(v["true"], par)
                cf1 = 1.0 if (m["n_true_containers"] == 0 and m["n_pred_containers"] == 0) else m["container_f1"]
                rows.append({"page": k[0], "device": k[1], "level": lv, "system": name, "n_elements": L,
                             "bin": density_bin(L), "n_removed": int(removed.sum()) if name.endswith("trust") else 0,
                             **{x: m[x] for x in ["edge_f1", "leaf_edge_f1", "container_match", "container_precision",
                                                  "edit_distance", "n_true_containers", "n_pred_containers"]},
                             "container_f1": cf1})
            if has_meta:
                trust_rows.append({"page": k[0], "device": k[1], "level": lv, "n_elements": L,
                                   "removed": int(removed.sum()), "trust_mean": float(t[:L].mean())})
    return pd.DataFrame(rows), pd.DataFrame(trust_rows)


def summarise(cfg, df, tr):
    nb = cfg["n_boot"]
    out = {"structure": {}, "paired": {}, "by_bin": {}, "removal": {}, "pages": {}}
    for (dev, lv, sysn), d in df.groupby(["device", "level", "system"]):
        out["structure"][f"{dev}|{lv}|{sysn}"] = {m: bootstrap_mean(d[m], nb, cfg["seed"]) for m in
                                                  ["container_f1", "edge_f1", "container_match", "container_precision",
                                                   "edit_distance"]}
        if sysn in ("fusion_full_s0+trust", "xycut"):
            out["by_bin"][f"{dev}|{lv}|{sysn}"] = {b: {"n": len(x), "container_f1": float(x.container_f1.mean())}
                                                   for b, x in d.groupby("bin")}
    for dev in df.device.unique():
        d = df[df.device == dev]
        out["pages"][dev] = {"n": int(d.page.nunique()), "elements_median": float(d.groupby("page").n_elements.first().median()),
                             "true_containers_mean": float(d.groupby("page").n_true_containers.first().mean())}
        for lv in d.level.unique():
            piv = d[d.level == lv].pivot(index="page", columns="system", values="container_f1")
            for a, b in cfg["pairs"]:
                if a in piv and b in piv:
                    out["paired"][f"{dev}|{lv}|{a} - {b}"] = paired_bootstrap(piv[a], piv[b], nb, cfg["seed"])
    for (dev, lv), d in tr.groupby(["device", "level"]):
        out["removal"][f"{dev}|{lv}"] = {"elements": int(d.n_elements.sum()), "removed": int(d.removed.sum()),
                                         "share_removed": float(d.removed.sum() / d.n_elements.sum()),
                                         "pages_with_removal": float((d.removed > 0).mean()),
                                         "trust_mean": float(d.trust_mean.mean())}
    return out


def stage_run(cfg, split, confirm):
    if split == "test" and not confirm:
        sys.exit("test runs once: pass --confirm-test")
    name = f"web_{split}"
    if split == "test" and any(Path(cfg["runs_root"]).glob(f"{name}_run*/metrics.json")):
        sys.exit("a finished web test run exists; the test is run once")
    run = start_run(name, cfg, cfg["runs_root"])
    t0 = time.time()
    data = pickle.load(open(cache_file(cfg, split), "rb"))
    items = data["items"]
    sysm = System(cfg)
    print(f"{len(items)} page-devices; t_c {sysm.t_c_plain} / with trust removal t_c {sysm.t_c}, t_e {sysm.t_e}", flush=True)
    emb = embed(sysm, cfg, split, items)
    t1 = time.time()
    df, tr = score_rows(cfg, sysm, items, emb)
    df.to_csv(run / "per_page.csv.gz", index=False)
    tr.to_csv(run / "removal_per_page.csv.gz", index=False)
    pd.DataFrame(data["infos"]).to_csv(run / "pages.csv", index=False)
    met = {"split": split, **summarise(cfg, df, tr),
           "sec": {"embed": round(t1 - t0, 1), "score": round(time.time() - t1, 1)},
           "thresholds": {"t_c": sysm.t_c_plain, "t_c_trust": sysm.t_c, "t_e": sysm.t_e}}
    finish_run(run, met)
    for k, v in met["structure"].items():
        c = v["container_f1"]
        print(f"{k:45s} container F1 {c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] n={c['n']}")
    for k, v in met["paired"].items():
        print(f"{k:60s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    print(json.dumps(met["removal"], indent=1))
    print("run folder:", run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluate_android_system_on_web_pages.yaml")
    ap.add_argument("--stage", required=True, choices=["fetch", "build", "run"])
    ap.add_argument("--split", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.stage == "fetch":
        stage_fetch(cfg)
    elif a.stage == "build":
        stage_build(cfg, a.split)
    else:
        stage_run(cfg, a.split, a.confirm_test)
