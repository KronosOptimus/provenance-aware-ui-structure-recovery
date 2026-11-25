"""Web-trained versions of the provenance-aware network, compared with the Android-only network.
Config: configs/train_web_only_and_joint_android_web_networks.yaml (web inputs as configs/evaluate_android_system_on_web_pages.yaml, network and Android data as
configs/train_trust_score_on_corrupted_metadata.yaml).

Regimes (each trained with 3 seeds; settings as the Android network, Model 4's selected trial)
  web     WebUI-7k training pages only; no trust head (no web element is known to be wrong, so the head
          would have nothing to learn); levels real / drop + mask (no injection) / absent
  joint   CLAY training screens (levels and injected fakes exactly as the Android network) + the same web pages;
          trust head kept: web elements are all faithful examples
  fusion_full  the Android-only network (3 seeds, already trained), for comparison

Stages
  fetch            WebUI training, validation and test archives, one at a time: page lists, needed files, page
                   addresses; training pages from a website that also appears in validation or test are dropped,
                   validation pages from a test website too
  build --split    each page and device -> network inputs at every level (val / test exactly as the web test)
  embed --split    crop embeddings with Model 3's frozen backbone
  train --regime r --seed s
  dev              predictions on the web validation pages; thresholds per system (web validation, real level;
                   Android validation for the Android test); validation report
  test             runs once: the 1,000 web test pages of the web test and the Android test
"""
import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_trust_score_on_corrupted_metadata as P10  # noqa: E402
import evaluate_android_system_on_web_pages as WT  # noqa: E402

from uistruct.data import provenance_tokens as PT  # noqa: E402
from uistruct.data import webui as W  # noqa: E402
from uistruct.data.nesting_crops import screen_boxes  # noqa: E402
from uistruct.eval.metrics import density_bin, grouping_metrics, smooth_single_child  # noqa: E402
from uistruct.eval.stats import bootstrap_mean, paired_bootstrap  # noqa: E402
from uistruct.runlog import finish_run, start_run  # noqa: E402
from uistruct.train import provenance_head as PH  # noqa: E402

DEVICE = P10.DEVICE
CROP_DIM = P10.CROP_DIM
LEVEL_NAMES = ["real", "heavy", "absent"]          # draw names; a web page's second level is drop + mask only
SECOND_LEVEL = "heavy_noinject"


# ---------------------------------------------------------------- config
def load_cfg(path, laptop=False, smoke=False):
    cfg = yaml.safe_load(open(path))
    if laptop or smoke:
        cfg.update(cfg["laptop"])
    if smoke:
        cfg.update(cfg["smoke"])
    cfg["is_smoke"] = bool(smoke)
    cfg["workers"] = cfg.get("workers") or os.cpu_count()
    c10 = P10.load_cfg(cfg["fusion_config"], laptop, smoke)
    wcfg = yaml.safe_load(open(cfg["web_test_config"]))
    wcfg["workers"] = cfg["workers"]
    return cfg, c10, wcfg


def _json(p, obj=None):
    p = Path(p)
    if obj is None:
        return json.loads(p.read_text())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float))
    return obj


# ---------------------------------------------------------------- fetch
class LocalZip:
    """A downloaded archive with the RemoteZip interface used by webui.fetch_pages."""

    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        self.info = {i.filename: i for i in self.z.infolist()}
        self.lock = threading.Lock()

    def member(self, name):
        with self.lock:
            return self.z.read(name)

    def close(self):
        self.z.close()


def open_archive(cfg, split):
    """Whole archive downloaded (Colab: two to five large requests) or read remotely by ranges (laptop check)."""
    if not cfg["download"]:
        return W.RemoteZip(split), None
    repo, stem, n = W.SPLITS[split]
    dst = Path(cfg["zip_dir"]) / f"{split}.zip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        t0, tmp = time.time(), dst.with_suffix(".part")
        tmp.unlink(missing_ok=True)
        for i in range(1, n + 1):
            part = dst.parent / f"{split}.{i:03d}"
            subprocess.run(["curl", "-sSL", "--retry", "10", "-o", str(part),
                            W.HF.format(repo=repo, name=f"{stem}.zip.{i:03d}")], check=True)
            with open(tmp, "ab") as out, open(part, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
            part.unlink()
        tmp.replace(dst)
        print(f"{split}: archive downloaded, {dst.stat().st_size / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)
    return LocalZip(dst), dst


def page_urls(z, pids):
    stem = next(iter(z.info)).split("/")[0]
    out = {}
    for pid in pids:
        for dev in W.DEVICES.values():
            name = f"{stem}/{pid}/{dev}-url.txt"
            if name in z.info:
                out[pid] = z.member(name).decode("utf-8", "replace").strip()
                break
    return out


SECOND_LEVEL_DOMAINS = {"co", "com", "org", "net", "gov", "edu", "ac", "ne", "or", "go", "gob", "nic"}


def site(url):
    """Registered website of an address (www. removed; two-part country endings such as co.uk kept)."""
    host = (urlparse(url).hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    p = host.split(".")
    if len(p) >= 3 and len(p[-1]) == 2 and p[-2] in SECOND_LEVEL_DOMAINS:
        return ".".join(p[-3:])
    return ".".join(p[-2:])


def stage_fetch(cfg):
    raw = Path(cfg["raw_dir"])
    test_pages = sorted(pd.read_csv(cfg["test_pages"], dtype={"page": str}).page.unique().tolist())
    assert len(test_pages) == 1000, "the web test has 1,000 pages"
    summ = {}
    for split in ["test", "val", "train"]:
        dst = raw / split
        if (dst / "urls.json").exists():
            print(f"{split}: already fetched", flush=True)
            summ[split] = _json(dst / "fetch.json")
            continue
        z, zpath = open_archive(cfg, split)
        pages = W.page_ids(z)
        ok = W.eligible(pages)
        rng = np.random.default_rng(cfg["seed"])
        if split == "test":
            pick = test_pages[:cfg["n_test_pages"]] if cfg.get("n_test_pages") else test_pages
            assert set(pick) <= set(ok), "test pages missing from the archive"
        elif split == "val":
            pick = sorted(rng.choice(ok, cfg["n_val_pages"], replace=False).tolist())
        else:
            pick = sorted(rng.choice(ok, cfg["n_train_pages"], replace=False).tolist()) if cfg.get("n_train_pages") else ok
        print(f"{split}: {len(pages)} pages in the archive, {len(ok)} with both devices complete; fetching {len(pick)}",
              flush=True)
        dst.mkdir(parents=True, exist_ok=True)
        _json(dst / "pages.json", pick)
        res = W.fetch_pages(z, pick, dst, cfg["fetch_threads"])
        urls = page_urls(z, pick)
        _json(dst / "fetch.json", {"archive_pages": len(pages), "eligible": len(ok), "picked": len(pick),
                                   "with_address": len(urls), **res})
        _json(dst / "urls.json", urls)
        summ[split] = _json(dst / "fetch.json")
        if zpath is not None:
            z.close()
            zpath.unlink()
    # website check: no website shared between training and validation / test, or validation and test
    urls = {s: _json(raw / s / "urls.json") for s in ["train", "val", "test"]}
    sites = {s: {p: site(u) for p, u in urls[s].items()} for s in urls}
    test_sites, val_sites = set(sites["test"].values()), set(sites["val"].values())
    pages = {s: _json(raw / s / "pages.json") for s in ["train", "val", "test"]}
    no_addr = {s: [p for p in pages[s] if p not in urls[s]] for s in pages}
    keep_train = [p for p in pages["train"] if p in sites["train"] and sites["train"][p] not in test_sites | val_sites]
    keep_val = [p for p in pages["val"] if p in sites["val"] and sites["val"][p] not in test_sites]
    _json(raw / "train" / "pages_kept.json", keep_train)
    _json(raw / "val" / "pages_kept.json", keep_val)
    check = {"pages": {s: len(pages[s]) for s in pages}, "no_address": {s: len(v) for s, v in no_addr.items()},
             "websites": {s: len(set(v.values())) for s, v in sites.items()},
             "train_pages_on_test_websites": sum(v in test_sites for v in sites["train"].values()),
             "train_pages_on_val_websites": sum(v in val_sites for v in sites["train"].values()),
             "val_pages_on_test_websites": sum(v in test_sites for v in sites["val"].values()),
             "shared_websites_train_test": sorted(set(sites["train"].values()) & test_sites)[:50],
             "train_kept": len(keep_train), "val_kept": len(keep_val), "fetch": summ}
    Path(cfg["cache_dir"]).mkdir(parents=True, exist_ok=True)
    _json(Path(cfg["cache_dir"]) / "fetch_check.json", check)
    print(json.dumps({k: v for k, v in check.items() if k != "fetch"}, indent=1), flush=True)


# ---------------------------------------------------------------- build
def _build_train_one(a):
    page_dir, device, page, wcfg, levels, seed = a
    g, raw, info = W.page_tables(page_dir, device, int(page), WT.flag_fill(wcfg))
    if g is None:
        return page, device, None, info
    fusion = yaml.safe_load(open(wcfg["fusion_config"]))
    sc = PT.build_screen_levels(g, raw, fusion["pool"], fusion["min_iou"], levels, {"p_inj": 0.0}, None, seed)
    info.update(n_leaves=len(sc["v"]["real"]["leaf_ids"]))
    return page, device, {"sc": sc}, info


def cache_file(cfg, split):
    return Path(cfg["cache_dir"]) / f"web_{split}.pkl"


def fingerprint(items):
    """sha256 over integers only (page, device, leaf ids, element kinds, candidate membership, labels) of every level."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(items):
        h.update(f"{k[0]}|{k[1]}".encode())
        for lv in sorted(items[k]["sc"]["v"]):
            v = items[k]["sc"]["v"][lv]
            h.update(lv.encode())
            h.update(np.asarray(v["leaf_ids"], np.int64).tobytes())
            h.update(v["leaf_kind"].astype(np.int8).tobytes())
            h.update(np.packbits(v["M"]["member"]).tobytes())
            h.update(v["M"]["y"].astype(np.int8).tobytes())
    return h.hexdigest()


def stage_build(cfg, wcfg, split):
    t0 = time.time()
    d = Path(cfg["raw_dir"]) / split
    pages = _json(d / ("pages.json" if split == "test" else "pages_kept.json"))
    if split == "train":
        fn = _build_train_one
        jobs = [(str(d / p), dv, p, wcfg, cfg["levels_train_web"], cfg["train_corruption_seed"])
                for p in pages for dv in wcfg["devices"]]
    else:                                   # exactly as the web test builds its pages (levels, seed, rule baselines)
        fn = WT._build_one
        jobs = [(str(d / p), dv, p, wcfg) for p in pages for dv in wcfg["devices"]]
    out, infos = {}, []
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        for n, (page, device, item, info) in enumerate(ex.map(fn, jobs, chunksize=8)):
            infos.append({"page": page, "device": device, **info})
            if item is not None:
                out[(page, device)] = item
            if (n + 1) % 2000 == 0:
                print(f"  {n + 1}/{len(jobs)} page views ({time.time() - t0:.0f}s)", flush=True)
    path = cache_file(cfg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"items": out, "infos": infos}, f, protocol=pickle.HIGHEST_PROTOCOL)
    inf = pd.DataFrame(infos)
    fp = fingerprint(out)
    summ = {"split": split, "pages": len(pages), "page_views": len(jobs), "kept": len(out), "fingerprint": fp,
            "by_device": {dv: {"kept": int(x.n_leaves.notna().sum()), "elements_median": float(x.n_leaves.median())}
                          for dv, x in inf.groupby("device")},
            "sec": round(time.time() - t0, 1)}
    _json(Path(cfg["cache_dir"]) / f"build_{split}.json", summ)
    print(json.dumps(summ, indent=1), flush=True)
    if split == "test" and not cfg["is_smoke"]:
        print("web test inputs identical to the web test's" if fp == cfg["test_fingerprint"]
              else "TEST INPUTS DIFFER FROM THE WEB TEST: stop and check", flush=True)


def load_items(cfg, split):
    with open(cache_file(cfg, split), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- embed
def stage_embed(cfg, c10, split):
    """Crop embeddings of every token box of every level (as the Android embeddings: GPU, float16 storage)."""
    t0 = time.time()
    items = load_items(cfg, split)["items"]
    keys = sorted(items)
    trunk = P10.model3_trunk(c10).to(DEVICE).eval()
    from uistruct.models.cnn import Normalise
    norm = Normalise().to(DEVICE)
    jobs, index, off = [], [], 0
    for k in keys:
        groups = {}
        for lv, v in items[k]["sc"]["v"].items():
            groups[f"{lv}|leaf"], groups[f"{lv}|M"] = v["leaf_xyxy"], v["M"]["crop_xyxy"]
        uniq, idx = screen_boxes(None, groups)
        jobs.append((str(Path(cfg["raw_dir"]) / split / k[0] / f"{k[1]}-screenshot.webp"), k[1], uniq))
        index.append({"key": k, "offset": off, **idx})
        off += len(uniq)
    emb = np.zeros((off, CROP_DIM), np.float16)
    row = 0
    with torch.no_grad(), ProcessPoolExecutor(cfg["workers"]) as ex:
        for w in range(0, len(jobs), 512):
            for crops in ex.map(WT._crop_one, jobs[w:w + 512], chunksize=4):
                for a in range(0, len(crops), 2048):
                    x = torch.from_numpy(crops[a:a + 2048]).to(DEVICE)
                    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                        e = trunk(norm(x))
                    emb[row:row + len(x)] = e.half().cpu().numpy()
                    row += len(x)
            print(f"  {min(w + 512, len(jobs))}/{len(jobs)} page views ({time.time() - t0:.0f}s)", flush=True)
    assert row == off
    np.save(Path(cfg["cache_dir"]) / f"emb_{split}.npy", emb)
    with open(Path(cfg["cache_dir"]) / f"emb_{split}_index.pkl", "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    summ = {"split": split, "page_views": len(keys), "boxes": off, "sec": round(time.time() - t0, 1)}
    _json(Path(cfg["cache_dir"]) / f"embed_{split}.json", summ)
    print(json.dumps(summ), flush=True)


def load_web_emb(cfg, split):
    e = np.load(Path(cfg["cache_dir"]) / f"emb_{split}.npy")
    idx = pickle.load(open(Path(cfg["cache_dir"]) / f"emb_{split}_index.pkl", "rb"))
    return e, {d["key"]: d for d in idx}


def web_crops_of(e, d, lv):
    o = d["offset"]
    return e[o + d[f"{lv}|leaf"]], e[o + d[f"{lv}|M"]]


def web_transform(enc, v, crops):
    """Web elements carry no type truth: the type loss ignores them."""
    x = enc.transform(v, crops)
    x["leaf_label"] = np.full(len(v["leaf_ids"]), -100, np.int64)
    return x


# ---------------------------------------------------------------- train
def model_dir(cfg, name):
    return Path(cfg["out_dir"]) / "models" / name


def stage_train(cfg, c10, regime, seed):
    spec = cfg["regimes"][regime]
    assert seed in spec["seeds"], "seed not in the plan for this regime"
    name = f"{regime}_s{seed}"
    d = model_dir(cfg, name)
    sel = spec.get("select_levels", ["real"])
    if (d / "trial.json").exists() and _json(d / "trial.json").get("select_levels", ["real"]) == sel:
        print(f"[{name}] already saved, skipped", flush=True)
        return
    t0 = time.time()
    hp = P10.model4_hp(c10)
    weights = dict(zip(LEVEL_NAMES, cfg["level_weights"]))
    web_lv = [cfg["levels_train_web"][LEVEL_NAMES.index(n)] for n in LEVEL_NAMES]
    wtr = load_items(cfg, "train")["items"]
    ew, iw = load_web_emb(cfg, "train")
    wkeys = sorted(wtr)
    src = [("web", k) for k in wkeys]
    if spec["android"]:
        atr = P10.load_cache(c10, "train")
        ea, ia = P10.load_emb(c10, "train")
        src = [("android", i) for i in range(len(atr))] + src

    def item(s, j, enc=None):
        """(level dict, crops) of training screen s at draw level j, or its transformed item when enc is given."""
        kind, k = s
        if kind == "android":
            sc, lv = atr[k], LEVEL_NAMES[j]
            v, c = sc["v"][lv], P10.level_crops(ea, ia[sc["screen_id"]], lv)
            return (v, c) if enc is None else enc.transform(v, c)
        lv = web_lv[j]
        v, c = wtr[k]["sc"]["v"][lv], web_crops_of(ew, iw[k], lv)
        return (v, c) if enc is None else web_transform(enc, v, c)

    # encoder statistics: one level per training screen, drawn once with the config seed (same for every seed)
    pick = PH.draw_levels(len(src), LEVEL_NAMES, weights, np.random.default_rng(cfg["seed"]))
    enc = PH.ProvenanceEncoder()
    drawn = [item(s, j) for s, j in zip(src, pick)]
    enc.fit([v for v, _ in drawn])
    enc.fit_crops([c[0] for _, c in drawn], [c[1] for _, c in drawn])
    del drawn
    tr_levels = [[item(s, j, enc) for j in range(len(LEVEL_NAMES))] for s in src]
    # validation (early stopping on candidate AP): web validation pages at the regime's selection levels (the web real
    # level alone is circular: the true tree comes from the same page tree), + Android validation, real level (joint)
    wdv = load_items(cfg, "val")["items"]
    evw, ivw = load_web_emb(cfg, "val")
    dvx = [web_transform(enc, wdv[k]["sc"]["v"][lv], web_crops_of(evw, ivw[k], lv)) for lv in sel for k in sorted(wdv)]
    if spec["android"]:
        dv = P10.load_cache(c10, "dev")
        edv, idv = P10.load_emb(c10, "dev")
        dvx += [enc.transform(s["v"]["real"], P10.level_crops(edv, idv[s["screen_id"]], "real")) for s in dv]
        del dv, edv
    n_and = sum(s[0] == "android" for s in src)
    print(f"[{name}] training on {len(src)} screens ({n_and} Android, {len(src) - n_and} web page views), "
          f"validation {len(dvx)}, trust head {spec['trust']}, settings {json.dumps(hp)}", flush=True)
    model, hist = PH.train_fusion(hp, enc, tr_levels, LEVEL_NAMES, weights, dvx, seed, CROP_DIM, spec["trust"],
                               c10["trust_weight"], device=DEVICE, workers=c10["loader_workers"])
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "hp": hp, "trust": spec["trust"],
                "variant": regime, "seed": seed}, d / "model.pt")
    pickle.dump(enc, open(d / "enc.pkl", "wb"))
    best = int(np.argmax([h["dev_ap"] for h in hist]))
    _json(d / "trial.json", {"name": name, "hp": hp, "weights": weights, "web_levels": web_lv, "trust": spec["trust"],
                             "select_levels": sel,
                             "android_screens": n_and, "web_page_views": len(src) - n_and, "history": hist,
                             "dev_ap": hist[best]["dev_ap"], "best_epoch": best + 1,
                             "minutes": round((time.time() - t0) / 60, 1)})
    print(f"[{name}] best epoch {best + 1}, validation AP {hist[best]['dev_ap']:.4f}, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- prediction
def all_models(cfg):
    """(name, model.pt folder) of every network compared: Android-only seeds, then the trained regimes."""
    out = [(m, Path(cfg["android_models_dir"]) / m) for m in cfg["android_models"]]
    for r, spec in cfg["regimes"].items():
        for s in spec["seeds"]:
            d = model_dir(cfg, f"{r}_s{s}")
            assert (d / "trial.json").exists(), f"{r}_s{s} not trained yet"
            out.append((f"{r}_s{s}", d))
    return out


def load_model(d):
    st = torch.load(Path(d) / "model.pt", map_location="cpu", weights_only=False)
    enc = pickle.load(open(Path(d) / "enc.pkl", "rb"))
    model = PH.build_fusion(st["hp"], enc, CROP_DIM, st["trust"])
    model.load_state_dict(st["state"])
    return model.to(DEVICE), enc, st


def predict(cfg, c10, models, vs_by_level, crops_by_level, web):
    """{(model, level): {"cand", "pinv", "trust"}} over the listed level dicts."""
    out = {}
    for name, d in models:
        model, enc, st = load_model(d)
        for lv, vs in vs_by_level.items():
            t0 = time.time()
            tf = web_transform if web else (lambda e, v, c: e.transform(v, c))
            items = [tf(enc, v, c) for v, c in zip(vs, crops_by_level[lv])]
            c, t, r = PH.predict_fusion(model, items, st["hp"]["budget"], DEVICE, c10["loader_workers"])
            out[(name, lv)] = {"cand": c, "pinv": [x[:, 0] for x in t], "trust": r if st["trust"] else None}
            print(f"  {name} {lv}: {len(items)} screens ({time.time() - t0:.0f}s)", flush=True)
        del model
    return out


def web_inputs(cfg, split, levels):
    data = load_items(cfg, split)
    items = data["items"]
    keys = sorted(items)
    e, idx = load_web_emb(cfg, split)
    vs = {lv: [items[k]["sc"]["v"][lv] for k in keys] for lv in levels}
    cr = {lv: [web_crops_of(e, idx[k], lv) for k in keys] for lv in levels}
    return items, keys, vs, cr, data["infos"]


def android_inputs(c10, split, levels):
    screens = P10.load_cache(c10, split)
    e, idx = P10.load_emb(c10, split)
    vs = {lv: [s["v"][lv] for s in screens] for lv in levels}
    cr = {lv: [P10.level_crops(e, idx[s["screen_id"]], lv) for s in screens] for lv in levels}
    return screens, vs, cr


def systems(preds):
    """(system, model, removal source) of every network system: nothing removed, and trust removal if a head."""
    out = []
    for m in sorted({m for m, _ in preds}):
        out.append((m, m, "none"))
        if preds[(m, "real")]["trust"] is not None:
            out.append((f"{m}+trust", m, "trust"))
    return out


def tune(c10, vs, preds, syss):
    """Plan §4 on the validation real level (the Android network's procedure), per system."""
    out = {}
    with ProcessPoolExecutor(c10["workers"]) as ex:
        for sysname, m, src in syss:
            t0 = time.time()
            p = preds[(m, "real")]
            prs = [{k: (p[k][i] if p[k] is not None else None) for k in ("cand", "pinv", "trust")} for i in range(len(vs))]
            out[sysname] = P10.tune_system(ex, c10, vs, prs, src, None)
            print(f"  [{sysname}] t_c {out[sysname]['t_c']} t_e {out[sysname]['t_e']} ({time.time() - t0:.0f}s)", flush=True)
    return out


def removed(v, trust, t_e):
    """Elements removed by trust (plan §7: never at the absent level, where no metadata claims anything)."""
    L = len(v["leaf_ids"])
    if t_e is None or trust is None or not v["has_meta"]:
        return np.zeros(L, bool)
    return trust[:L] < t_e


def regime(sysname):
    return re.sub(r"_s\d+", "", sysname)


# ---------------------------------------------------------------- web scoring
def web_rows(items, keys, vs, preds, choice):
    """Container F1 etc. per page view, level and system. choice: {system: (model, t_c, t_e)}."""
    rows = []
    for lv in vs:
        for i, k in enumerate(keys):
            v = vs[lv][i]
            L = len(v["leaf_ids"])
            par = {}
            for sname, (m, tc, te) in choice.items():
                p = preds[(m, lv)]
                rm = removed(v, p["trust"][i] if p["trust"] is not None else None, te)
                par[sname] = (PH.removal_tree(v, p["cand"][i], tc, rm), int(rm.sum()))
            rules = items[k]["rules"]
            par["xycut"], par["flat"] = (rules["xycut"], 0), (rules["flat"], 0)
            if f"raw_collapsed|{lv}" in rules:
                par["raw_collapsed"] = (rules[f"raw_collapsed|{lv}"], 0)
            for sname, (pred, nrm) in par.items():
                mm = grouping_metrics(v["true"], pred)
                rows.append({"page": k[0], "device": k[1], "level": lv, "system": sname, "regime": regime(sname),
                             "n_elements": L, "bin": density_bin(L), "n_removed": nrm,
                             **{x: mm[x] for x in ["edge_f1", "container_match", "edit_distance", "n_true_containers",
                                                   "n_pred_containers"]},
                             "container_f1": P10.cf1_rule(mm)})
    return pd.DataFrame(rows)


def summarise(df, group, unit, pairs, n_boot, seed):
    """Per system and per regime (seed-averaged per page); seed spread; paired regime differences."""
    out = {"system": {}, "regime": {}, "seed_spread": {}, "paired": {}}
    for keys, x in df.groupby(group + ["system"]):
        out["system"]["|".join(keys)] = {"container_f1": bootstrap_mean(x.container_f1, n_boot, seed),
                                         "edge_f1": bootstrap_mean(x.edge_f1, n_boot, seed),
                                         "mean_removed": float(x.n_removed.mean())}
    avg = df.groupby(group + ["regime", unit]).container_f1.mean().reset_index()
    for keys, x in avg.groupby(group + ["regime"]):
        out["regime"]["|".join(keys)] = bootstrap_mean(x.container_f1, n_boot, seed)
    for keys, x in df.groupby(group + ["regime"]):
        means = x.groupby("system").container_f1.mean()
        out["seed_spread"]["|".join(keys)] = {"seeds": int(len(means)), "mean": float(means.mean()),
                                              "sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                                              "per_seed": means.round(4).to_dict()}
    for keys, x in avg.groupby(group):
        keys = keys if isinstance(keys, tuple) else (keys,)
        piv = x.pivot(index=unit, columns="regime", values="container_f1")
        for a, b in pairs:
            if a in piv and b in piv:
                out["paired"]["|".join(keys) + f"|{a} - {b}"] = paired_bootstrap(piv[a], piv[b], n_boot, seed)
    return out


def android_thresholds(cfg):
    tuned = _json(cfg["android_test_run"])["tuned"]
    return {m: {"t_c": tuned[m]["t_c"], "t_e": tuned[m]["t_e"]} for m in tuned}


def web_choice(models_preds, tuned, and_thr):
    """Systems scored on web pages: each network with web-validation thresholds; the Android-only network also with
    its Android-validation thresholds (as the web test ran it)."""
    choice = {}
    for sysname, m, src in systems(models_preds):
        t = tuned[sysname]
        choice[sysname] = (m, t["t_c"], t["t_e"] if src == "trust" else None)
        if sysname in and_thr:
            a = and_thr[sysname]
            choice[f"{sysname}@android_thr"] = (m, a["t_c"], a["t_e"] if src == "trust" else None)
    return choice


def print_web(met, title):
    print(f"===== {title}: container F1, seed-averaged per page view, mean [95% CI]")
    for k, v in sorted(met["regime"].items()):
        print(f"{k:50s} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] n={v['n']}")
    print("----- paired (seed-averaged)")
    for k, v in met["paired"].items():
        print(f"{k:70s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")


# ---------------------------------------------------------------- Android scoring
def _arows_job(a):
    v, scores, tc, rm, sysname, lv = a
    pred = PH.removal_tree(v, scores, tc, rm)
    base = {"screen_id": v["screen_id"], "system": sysname, "regime": regime(sysname), "level": lv,
            "n_elements": len(v["leaf_ids"]), "n_removed": int(rm.sum())}
    rows = []
    for tname in ("raw", "smoothed"):
        tt = v["true"] if tname == "raw" else smooth_single_child(v["true"])
        mm = grouping_metrics(tt, pred)
        rows.append({**base, "truth": tname, "edge_f1": mm["edge_f1"], "container_f1": P10.cf1_rule(mm)})
    return rows


def android_rows(c10, vs, preds, choice):
    rows = []
    with ProcessPoolExecutor(c10["workers"]) as ex:
        for sname, (m, tc, te) in choice.items():
            t0 = time.time()
            for lv in vs:
                p = preds[(m, lv)]
                jobs = [(v, p["cand"][i], tc, removed(v, p["trust"][i] if p["trust"] is not None else None, te), sname, lv)
                        for i, v in enumerate(vs[lv])]
                rows += [r for rr in ex.map(_arows_job, jobs, chunksize=32) for r in rr]
            print(f"  [{sname}] scored ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- dev / test
def stage_dev(cfg, c10, wcfg):
    run = start_run("web_train_dev", cfg, cfg["runs_root"])
    t0 = time.time()
    models = all_models(cfg)
    levels = wcfg["levels"]
    items, keys, vs, cr, _ = web_inputs(cfg, "val", levels)
    print(f"web validation: {len(keys)} page views", flush=True)
    wp = predict(cfg, c10, models, vs, cr, True)
    print("thresholds on web validation (real level):", flush=True)
    wt = tune(c10, vs["real"], wp, systems(wp))
    and_thr = android_thresholds(cfg)
    df = web_rows(items, keys, vs, wp, web_choice(wp, wt, and_thr))
    df.to_csv(run / "web_val_per_page.csv.gz", index=False)
    wmet = summarise(df, ["device", "level"], "page", cfg["pairs_web"], cfg["n_boot"], cfg["seed"])
    # Android validation thresholds for the networks trained here (the Android-only network keeps its own choices)
    screens, avs, acr = android_inputs(c10, "dev", ["real"])
    new = [(n, d) for n, d in models if n not in cfg["android_models"]]
    ap = predict(cfg, c10, new, avs, acr, False)
    print("thresholds on Android validation (real level):", flush=True)
    at = tune(c10, avs["real"], ap, systems(ap))
    at.update({m: and_thr[m] for s in cfg["android_models"] for m in (s, f"{s}+trust")})
    choice = {"web": {k: {"t_c": v["t_c"], "t_e": v["t_e"]} for k, v in wt.items()},
              "android": {k: {"t_c": v["t_c"], "t_e": v["t_e"]} for k, v in at.items()}}
    _json(Path(cfg["out_dir"]) / "dev_choice.json", choice)
    trials = {n: {k: _json(d / "trial.json")[k] for k in ("dev_ap", "best_epoch", "minutes")}
              for n, d in models if n not in cfg["android_models"]}
    finish_run(run, {"split": "val", "web": wmet, "choice": choice, "tuning_web": wt, "tuning_android": at,
                     "trials": trials, "page_views": len(keys), "sec": round(time.time() - t0, 1)})
    print(json.dumps(trials, indent=1))
    print(json.dumps(choice, indent=1))
    print_web(wmet, "web validation")
    print("run folder:", run)


def stage_test(cfg, c10, wcfg, confirm):
    if not confirm:
        sys.exit("test runs once: pass --confirm-test")
    done = Path(cfg["out_dir"]) / "TEST_DONE"
    if done.exists():
        sys.exit("the test has been run; it runs once")
    choice = _json(Path(cfg["out_dir"]) / "dev_choice.json")
    done.write_text("started")
    run = start_run("web_train_test", cfg, cfg["runs_root"])
    t0 = time.time()
    models = all_models(cfg)
    levels = wcfg["levels"]
    items, keys, vs, cr, infos = web_inputs(cfg, "test", levels)
    print(f"web test: {len(keys)} page views", flush=True)
    wp = predict(cfg, c10, models, vs, cr, True)
    df = web_rows(items, keys, vs, wp, web_choice(wp, choice["web"], android_thresholds(cfg)))
    df.to_csv(run / "web_test_per_page.csv.gz", index=False)
    pd.DataFrame(infos).to_csv(run / "web_test_pages.csv", index=False)
    wmet = summarise(df, ["device", "level"], "page", cfg["pairs_web"], cfg["n_boot"], cfg["seed"])
    del wp
    t1 = time.time()
    screens, avs, acr = android_inputs(c10, "test", c10["levels_eval"])
    print(f"Android test: {len(screens)} screens", flush=True)
    ap = predict(cfg, c10, models, avs, acr, False)
    ach = {s: (m, choice["android"][s]["t_c"], choice["android"][s]["t_e"] if src == "trust" else None)
           for s, m, src in systems(ap)}
    adf = android_rows(c10, avs, ap, ach)
    adf.to_csv(run / "android_test_per_screen.csv.gz", index=False)
    amet = summarise(adf, ["level", "truth"], "screen_id", cfg["pairs_android"], cfg["n_boot"], cfg["seed"])
    finish_run(run, {"split": "test", "web": wmet, "android": amet, "choice": choice,
                     "sec": {"web": round(t1 - t0, 1), "android": round(time.time() - t1, 1)}})
    done.write_text("done")
    print_web(wmet, "web test")
    print_web(amet, "Android test")
    print("run folder:", run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_web_only_and_joint_android_web_networks.yaml")
    ap.add_argument("--stage", required=True, choices=["fetch", "build", "embed", "train", "dev", "test"])
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--regime")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--laptop", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    cfg, c10, wcfg = load_cfg(a.config, a.laptop, a.smoke)
    if a.stage == "fetch":
        stage_fetch(cfg)
    elif a.stage == "build":
        stage_build(cfg, wcfg, a.split)
    elif a.stage == "embed":
        stage_embed(cfg, c10, a.split)
    elif a.stage == "train":
        stage_train(cfg, c10, a.regime, a.seed)
    elif a.stage == "dev":
        stage_dev(cfg, c10, wcfg)
    else:
        stage_test(cfg, c10, wcfg, a.confirm_test)
