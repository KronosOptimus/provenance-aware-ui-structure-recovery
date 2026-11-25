"""Provenance network: wrong metadata, trust head and calibration.
Config: configs/train_trust_score_on_corrupted_metadata.yaml.

Stages
  rates        (laptop, train + dev check) how many wrong boxes the metadata delivers; sets p_inj for injection
  fingerprint  (laptop, train only) can a classifier tell injected fakes from real invalid nodes? (pass rule in config)
  build  --split train|dev|test   every corruption level of every screen -> cache (train: training levels;
                                  dev/test: stress levels, own seeds)
  baselines    (laptop) p(invalid) of Models 2-3 etc. per validation / test element -> one small file for Drive
  embed  --split ...              crop embeddings of every box of every level (Model 3's frozen backbone)
  train  --variant v --seed s     one provenance network (Model 4's selected settings + trust head)
  dev    validation report: assembly thresholds, element removal by each score, trust metrics and calibration
  test   ONCE (--confirm-test): the dev stage's choices on the clean test split
--laptop  laptop paths (config `laptop` block)   --smoke  first N screens only, not results
"""
import argparse
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from uistruct.data import nesting_crops as NC
from uistruct.data import provenance_tokens as PT
from uistruct.data.features import FLAGS
from uistruct.eval import trust as TR
from uistruct.eval.metrics import density_bin, grouping_metrics, smooth_single_child
from uistruct.eval.stats import bootstrap_mean, bootstrap_rows, paired_bootstrap
from uistruct.runlog import finish_run, start_run
from uistruct.train import provenance_head as PH
from uistruct.train.nesting_head import cand_ap

BOX = ["nx0", "ny0", "nx1", "ny1"]
RAW_COLS = ["screen_id", "node_idx", "parent_idx", "box_ok", *BOX, "depth", "n_children", "class_name", *FLAGS]
SPLIT_SEED = {"train": 0, "dev": 1, "test": 2}


# ---------------------------------------------------------------- config and data
def load_cfg(path, laptop, smoke):
    cfg = yaml.safe_load(open(path))
    if laptop or smoke:
        cfg.update(cfg["laptop"])
    if smoke:
        cfg.update(cfg["smoke"])
    cfg["is_smoke"] = bool(smoke)
    cfg["workers"] = cfg.get("workers") or os.cpu_count()
    return cfg


def sfx(cfg):
    return "_smoke" if cfg["is_smoke"] else ""


def load_tables(cfg, split, limit=None, sample=None, seed=0):
    """Valid elements grouped by screen and raw nodes (with CLAY label, for truth only) by screen.
    The raw nodes' CLAY label comes from the elements table (every labelled node of the clean splits, valid and
    invalid; only ROOT labels are missing, and ROOT carries no trust truth either way). The Colab tables have no
    label column on nodes; on the laptop this gives the same values as nodes.parquet's own label."""
    el_all = pd.read_parquet(f"{cfg['data_dir']}/elements.parquet",
                             columns=["screen_id", "node_idx", "gt_parent", "label", "is_valid", "split_clean", *BOX])
    el_all = el_all[el_all.split_clean == split]
    el = el_all[el_all.is_valid]
    sids = sorted(el.screen_id.unique())
    if sample:
        sids = sorted(np.random.default_rng(seed).choice(sids, min(sample, len(sids)), replace=False).tolist())
    if limit:
        sids = sids[:limit]
    el = el[el.screen_id.isin(set(sids))]
    nodes = pd.read_parquet(f"{cfg['data_dir']}/nodes.parquet", columns=RAW_COLS,
                            filters=[("screen_id", "in", sids)])
    nodes = nodes.merge(el_all[["screen_id", "node_idx", "label"]], on=["screen_id", "node_idx"], how="left",
                        validate="one_to_one")
    nodes["is_labelled"] = nodes.label.notna()
    nodes["label"] = nodes.label.fillna(-1).astype(np.int8)
    del el_all
    raw_by = {k: v.sort_values("node_idx").reset_index(drop=True) for k, v in nodes.groupby("screen_id", sort=False)}
    groups = dict(tuple(el.groupby("screen_id", sort=True)))
    return sids, groups, raw_by, nodes


def p_inj(cfg):
    return json.load(open(cfg["rates_run"]))["train"]["p_inj"]


# ---------------------------------------------------------------- rates (laptop)
def _rate_one(args):
    g, raw = args
    true = dict(zip(g.node_idx.tolist(), g.gt_parent.tolist()))
    vleaves = set(true) - set(true.values())
    inv_el = PT.invalid_elements(raw, vleaves)
    inv_all = int(((raw.label == 0) & raw.is_labelled & raw.box_ok).sum())
    return len(vleaves), len(inv_el), inv_all


def stage_rates(cfg):
    out = {}
    for split in ["train", "dev"]:
        t0 = time.time()
        sids, groups, raw_by, _ = load_tables(cfg, split, cfg.get("limit"))
        with ProcessPoolExecutor(cfg["workers"]) as ex:
            r = np.array(list(ex.map(_rate_one, [(groups[s], raw_by[s]) for s in sids], chunksize=64)))
        v, e, a = r[:, 0], r[:, 1], r[:, 2]
        share = e / (v + e)
        out[split] = {"n_screens": len(sids), "valid_leaves": int(v.sum()), "invalid_elements": int(e.sum()),
                      "invalid_usable_box_total": int(a.sum()),
                      "invalid_held_as_candidates_or_wrappers": int(a.sum() - e.sum()),
                      "p_inj": float(e.sum() / v.sum()),
                      "invalid_share_of_elements": float(e.sum() / (v.sum() + e.sum())),
                      "screens_with_invalid_element": float((e > 0).mean()),
                      "per_screen_invalid_elements": {"mean": float(e.mean()), "median": float(np.median(e)),
                                                      "p90": float(np.quantile(e, 0.9)), "max": int(e.max())},
                      "per_screen_invalid_share_if_any": {"median": float(np.median(share[e > 0])),
                                                          "p90": float(np.quantile(share[e > 0], 0.9))},
                      "sec": round(time.time() - t0, 1)}
        print(split, json.dumps(out[split]), flush=True)
    out["note"] = "p_inj = invalid elements / valid leaves on train; a heavy screen gets about twice its real errors"
    return out


# ---------------------------------------------------------------- build
def _build_one(args):
    g, raw, cfg, levels, rates, seed = args
    return PT.build_screen_levels(g, raw, cfg["pool"], cfg["min_iou"], levels, rates, _DONORS, seed)


_DONORS = None


def _init_donors(d):
    global _DONORS
    _DONORS = d


def _invalid_one(args):
    g, raw = args
    true = dict(zip(g.node_idx.tolist(), g.gt_parent.tolist()))
    return PT.invalid_elements(raw, set(true) - set(true.values()))


def split_donors(cfg, split):
    """Donors = the invalid elements of every screen of the split (not only the sampled ones)."""
    sids, groups, raw_by, nodes = load_tables(cfg, split, cfg.get("limit"))
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        inv = dict(zip(sids, ex.map(_invalid_one, [(groups[s], raw_by[s]) for s in sids], chunksize=64)))
    return PT.donor_table(nodes, inv)


def build_split(cfg, split, levels, sids=None, sample=None, seed_offset=0):
    donors = split_donors(cfg, split)
    sids_all, groups, raw_by, nodes = load_tables(cfg, split, cfg.get("limit"), sample, cfg["seed"])
    del nodes
    rates = {"p_inj": p_inj(cfg)}
    seed = cfg["corruption_seed"] + SPLIT_SEED[split] + seed_offset
    out, t0 = [], time.time()
    with ProcessPoolExecutor(cfg["workers"], initializer=_init_donors, initargs=(donors,)) as ex:
        for w in range(0, len(sids_all), 1024):
            win = sids_all[w:w + 1024]
            out.extend(ex.map(_build_one, [(groups[s], raw_by[s], cfg, levels, rates, seed) for s in win], chunksize=16))
            print(f"  {min(w + 1024, len(sids_all))}/{len(sids_all)} screens ({time.time() - t0:.0f}s)", flush=True)
    return out


def cache_path(cfg, split):
    return Path(cfg["cache_dir"]) / f"fusion_{split}{sfx(cfg)}.pkl"


def stage_build(cfg, split):
    t0 = time.time()
    levels = cfg["levels_train"] if split == "train" else cfg["levels_eval"]
    out = build_split(cfg, split, levels)
    path = cache_path(cfg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    summ = {"split": split, "n_screens": len(out), "levels": levels, "sec": round(time.time() - t0, 1),
            "fingerprint": cache_fingerprint(out)}
    for lv in levels:
        vs = [s["v"][lv] for s in out]
        k = np.concatenate([v["leaf_kind"] for v in vs])
        summ[lv] = {"elements": int(len(k)), "valid": int((k == 0).sum()), "clay_invalid": int((k == 1).sum()),
                    "injected": int((k == 2).sum()), "masked": int(sum(v["leaf_meta"][:, -1].sum() for v in vs)),
                    "mean_candidates": float(np.mean([len(v["M"]["sets"]) for v in vs])),
                    "mean_positives": float(np.mean([v["M"]["y"].sum() for v in vs])),
                    "cand_trust_labelled": int(sum((v["M"]["trust"] >= 0).sum() for v in vs))}
    return summ


# ---------------------------------------------------------------- fingerprint check (laptop, train only)
def stage_fingerprint(cfg):
    """Injected fakes vs real CLAY invalid elements on the heavy level of sampled train screens.
    Features = what the model sees of an element: box, metadata (not the mask flag, which is drawn
    independently of validity), class name (frequency-coded), Model 2 colour statistics of the crop."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from uistruct.data.colour import COLOUR, screen_features
    fp = cfg["fingerprint"]
    t0 = time.time()
    scr = build_split(cfg, "train", ["heavy"], sample=fp["n_screens"])
    rows = []
    for s in scr:
        v = s["v"]["heavy"]
        keep = v["leaf_kind"] > 0
        if not keep.any():
            continue
        col = screen_features(f"{cfg['img_dir']}/{s['screen_id']}.jpg", v["leaf_xyxy"][keep])
        for i, j in enumerate(np.nonzero(keep)[0]):
            rows.append({"screen_id": s["screen_id"], "fake": int(v["leaf_kind"][j] == 2), "name": v["leaf_name"][j],
                         "masked": v["leaf_meta"][j, -1],
                         **{f"b{k}": x for k, x in enumerate(v["leaf_box"][j])},
                         **{f"m{k}": x for k, x in enumerate(v["leaf_meta"][j, :-1])},
                         **dict(zip(COLOUR, col[i]))})
    df = pd.DataFrame(rows)
    df = df[df.masked == 0]                         # masked elements show no metadata; compare what is visible
    cnt = df.name.value_counts()
    df["name_code"] = df.name.map(lambda n: cnt.get(n, 0)).astype(float)
    groups = {"box": [c for c in df if c.startswith("b")], "meta": [c for c in df if c.startswith("m")] + ["name_code"],
              "colour": list(COLOUR)}
    groups["all"] = groups["box"] + groups["meta"] + groups["colour"]
    sid = np.array(sorted(df.screen_id.unique()))
    test_s = set(np.random.default_rng(cfg["seed"]).choice(sid, len(sid) * 3 // 10, replace=False).tolist())
    te = df.screen_id.isin(test_s).to_numpy()
    res = {"n_real_invalid": int((df.fake == 0).sum()), "n_fake": int((df.fake == 1).sum()), "n_screens": len(sid)}
    for gname, cols in groups.items():
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=cfg["seed"])
        clf.fit(df.loc[~te, cols], df.fake[~te])
        res[f"auroc_{gname}"] = float(roc_auc_score(df.fake[te], clf.predict_proba(df.loc[te, cols])[:, 1]))
    res["pass"] = res["auroc_all"] <= fp["pass_auroc"]
    means = df.groupby("fake")[groups["all"]].mean().T
    means["diff_in_sd"] = (means[1] - means[0]) / df[groups["all"]].std().replace(0, np.nan)
    res["largest_mean_differences"] = means.diff_in_sd.abs().sort_values(ascending=False).head(8).round(3).to_dict()
    res["sec"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------- shared helpers (Colab stages)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROP_DIM = 512
METRICS = ["container_f1", "container_match", "container_precision", "edge_f1", "leaf_edge_f1", "edit_distance"]


def load_cache(cfg, split):
    with open(cache_path(cfg, split), "rb") as f:
        return pickle.load(f)


def cache_fingerprint(screens):
    """sha256 over integers only (ids, element kinds, candidate membership, labels) of every level."""
    h = hashlib.sha256()
    for s in screens:
        h.update(np.int64(s["screen_id"]).tobytes())
        for lv in sorted(s["v"]):
            v = s["v"][lv]
            h.update(lv.encode())
            h.update(np.asarray(v["leaf_ids"], np.int64).tobytes())
            h.update(v["leaf_kind"].astype(np.int8).tobytes())
            h.update(np.packbits(v["M"]["member"]).tobytes())
            h.update(v["M"]["y"].astype(np.int8).tobytes())
    return h.hexdigest()


def out_dir(cfg):
    p = Path(cfg["out_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json(p, obj=None):
    if obj is None:
        return json.load(open(p))
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(p, "w"), indent=1, default=float)
    return obj


# ---------------------------------------------------------------- baselines (laptop): p(invalid) of Models 2-3 etc.
def type_row_order(split, data_dir):
    """(screen_id, node_idx, label) in the row order of features.load_elements (Models 1-3 outputs); same steps
    as scripts/analyse_saved_outputs.type_row_order (the multi-column sort is stable, so one split sorts like all)."""
    el = pd.read_parquet(f"{data_dir}/elements.parquet", columns=["screen_id", "node_idx", "label", "is_valid", "split_clean"])
    el = el[el.split_clean.isin((split,))]
    nodes = pd.read_parquet(f"{data_dir}/nodes.parquet", columns=["screen_id", "node_idx", *BOX],
                            filters=[("screen_id", "in", sorted(el.screen_id.unique().tolist()))])
    df = el.merge(nodes, on=["screen_id", "node_idx"], how="left", validate="one_to_one")
    b = df[BOX].to_numpy(np.float64).clip(-0.5, 1.5)
    df["cx"], df["cy"] = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    return df.sort_values(["screen_id", "cy", "cx"]).reset_index(drop=True)[["screen_id", "node_idx", "label"]]


def stage_baselines(cfg):
    """One small file (uploaded to Drive): per validation / test element, p(invalid) of the earlier type models.
    Validation: Model 2, trees on Model 2 inputs, box + colour. Test: those + Model 3 and box + CNN.
    Model 3 / box + CNN validation probabilities are read on Colab from Drive (stage dev)."""
    b = cfg["baselines"]
    out = {}
    for split in ["dev", "test"]:
        keys = type_row_order(split, cfg["data_dir"])
        out[f"{split}_screen_id"] = keys.screen_id.to_numpy(np.int32)
        out[f"{split}_node_idx"] = keys.node_idx.to_numpy(np.int32)
        out[f"{split}_label"] = keys.label.to_numpy(np.int8)
        z = np.load(b[f"element_types_colour_{split}"])
        assert (z["screen_id"] == keys.screen_id.to_numpy()).all() and (z["y"] == keys.label.to_numpy()).all(), split
        for name, k in b["element_types_colour_keys"].items():
            out[f"{split}_{name}"] = z[k][:, 0].astype(np.float32)
        if split == "test":
            z8 = np.load(b["element_types_cnn_test"])
            assert (z8["screen_id"] == keys.screen_id.to_numpy()).all() and (z8["y"] == keys.label.to_numpy()).all()
            for name, k in b["element_types_cnn_keys"].items():
                out[f"test_{name}"] = z8[k][:, 0].astype(np.float32)
    Path(b["out"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(b["out"], **out)
    return {"file": b["out"], "mb": round(os.path.getsize(b["out"]) / 1e6, 2),
            "rows": {s: int(len(out[f"{s}_screen_id"])) for s in ("dev", "test")},
            "sources": sorted(k for k in out if not k.endswith(("_screen_id", "_node_idx", "_label")))}


def baseline_scores(cfg, split):
    """{source: {(screen_id, node_idx): p(faithful) = 1 - p(invalid)}}."""
    z = np.load(cfg["baselines"]["file"])
    keys = list(zip(z[f"{split}_screen_id"].tolist(), z[f"{split}_node_idx"].tolist()))
    src = {k[len(split) + 1:]: z[k] for k in z.files
           if k.startswith(split + "_") and k[len(split) + 1:] not in ("screen_id", "node_idx", "label")}
    if split == "dev":          # Model 3 and box + CNN validation probabilities: Model 3 checkpoints on Drive (Model 3 row order)
        for name, f in cfg["baselines"]["element_types_cnn_dev_files"].items():
            path = Path(cfg["element_types_cnn_ckpt_dir"]) / f
            if path.exists():
                a = np.load(path)
                assert len(a) == len(keys), f
                src[name] = a[:, 0].astype(np.float32)
            else:
                print(f"  baseline {name}: {path} not found, left out", flush=True)
    return {n: dict(zip(keys, (1.0 - v.astype(np.float64)).tolist())) for n, v in src.items()}


# ---------------------------------------------------------------- embed (Colab GPU)
def level_groups(s):
    g = {}
    for lv, v in s["v"].items():
        g[f"{lv}|leaf"], g[f"{lv}|M"] = v["leaf_xyxy"], v["M"]["crop_xyxy"]
    return g


def model3_trunk(cfg):
    from uistruct.models.cnn import resnet18_trunk
    if cfg.get("untrained_trunk"):                 # smoke only
        return resnet18_trunk(pretrained=False)
    st = torch.load(cfg["model3_ckpt"], map_location="cpu", weights_only=False)
    assert st["fine_tune"] and abs(st["hp"]["lr_backbone"] - cfg["model3_lr_backbone"]) < 1e-12, "not the selected Model 3"
    m = resnet18_trunk(pretrained=False)
    m.load_state_dict({k[len("backbone."):]: v for k, v in st["state"].items() if k.startswith("backbone.")})
    return m


def stage_embed(cfg, split):
    t0 = time.time()
    screens = load_cache(cfg, split)
    out, index = NC.embed_screens(screens, cfg["img_dir"], {"model3": model3_trunk(cfg)}, DEVICE, cfg["workers"],
                                  groups_fn=level_groups)
    os.makedirs(cfg["emb_dir"], exist_ok=True)
    NC.save(out, index, cfg["emb_dir"], split)
    n = len(out["model3"])
    return {"split": split, "n_screens": len(screens), "n_boxes": n, "boxes_per_screen": n / len(screens),
            "sec": round(time.time() - t0, 1)}


def load_emb(cfg, split):
    e, idx = NC.load(cfg["emb_dir"], split, "model3")
    return np.asarray(e), idx


def level_crops(e, idx, lv):
    o = idx["offset"]
    return np.asarray(e[o + idx[f"{lv}|leaf"]]), np.asarray(e[o + idx[f"{lv}|M"]])


# ---------------------------------------------------------------- train (Colab GPU)
def model4_hp(cfg):
    r = _json(cfg["model4_tune_metrics"])
    assert r["best"] == cfg["model4_trial"], "Model 4's selected trial changed"
    return {**r["trials"][r["best"]]["hp"], **cfg.get("hp_override", {})}


def model_dir(cfg, name):
    return out_dir(cfg) / "models" / name


def stage_train(cfg, a, run_dir):
    var = cfg["variants"][a.variant]
    assert a.seed in var["seeds"], "seed not in the plan for this variant"
    name = f"{a.variant}_s{a.seed}"
    d = model_dir(cfg, name)
    if (d / "trial.json").exists():
        print(f"[{name}] already saved, skipped", flush=True)
        return _json(d / "trial.json")
    t0 = time.time()
    hp = model4_hp(cfg)
    levels = list(var["weights"])
    tr, dv = load_cache(cfg, "train"), load_cache(cfg, "dev")
    etr, itr = load_emb(cfg, "train")
    # encoder statistics: one level per training screen, drawn once with the config seed (same for every seed)
    pick = PH.draw_levels(len(tr), levels, var["weights"], np.random.default_rng(cfg["seed"]))
    enc = PH.ProvenanceEncoder().fit([s["v"][levels[k]] for s, k in zip(tr, pick)])
    c = [level_crops(etr, itr[s["screen_id"]], levels[k]) for s, k in zip(tr, pick)]
    enc.fit_crops([x[0] for x in c], [x[1] for x in c])
    del c
    tr_levels = [[enc.transform(s["v"][lv], level_crops(etr, itr[s["screen_id"]], lv)) for lv in levels] for s in tr]
    del tr, etr
    edv, idv = load_emb(cfg, "dev")
    dvx = [enc.transform(s["v"]["real"], level_crops(edv, idv[s["screen_id"]], "real")) for s in dv]
    print(f"[{name}] training on {len(tr_levels)} screens, levels {var['weights']}, settings {json.dumps(hp)}", flush=True)
    model, hist = PH.train_fusion(hp, enc, tr_levels, levels, var["weights"], dvx, a.seed, CROP_DIM, var["trust"],
                               cfg["trust_weight"], device=DEVICE, workers=cfg["loader_workers"])
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "hp": hp, "trust": var["trust"],
                "variant": a.variant, "seed": a.seed}, d / "model.pt")
    pickle.dump(enc, open(d / "enc.pkl", "wb"))
    best = int(np.argmax([h["dev_ap"] for h in hist]))
    return _json(d / "trial.json", {"name": name, "hp": hp, "weights": var["weights"], "trust": var["trust"],
                                    "history": hist, "dev_ap": hist[best]["dev_ap"], "best_epoch": best + 1,
                                    "minutes": round((time.time() - t0) / 60, 1)})


def trained(cfg):
    return [f"{v}_s{s}" for v, spec in cfg["variants"].items() for s in spec["seeds"]
            if (model_dir(cfg, f"{v}_s{s}") / "trial.json").exists()]


def predict_split(cfg, split, screens, levels):
    """{(model, level): {"cand", "pinv", "trust", "cand_ap"}}; lists in screen order."""
    e, idx = load_emb(cfg, split)
    out = {}
    for name in trained(cfg):
        d = model_dir(cfg, name)
        st = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        enc = pickle.load(open(d / "enc.pkl", "rb"))
        model = PH.build_fusion(st["hp"], enc, CROP_DIM, st["trust"])
        model.load_state_dict(st["state"])
        model.to(DEVICE)
        for lv in levels:
            t0 = time.time()
            items = [enc.transform(s["v"][lv], level_crops(e, idx[s["screen_id"]], lv)) for s in screens]
            c, t, r = PH.predict_fusion(model, items, st["hp"]["budget"], DEVICE, cfg["loader_workers"])
            out[(name, lv)] = {"cand": c, "pinv": [x[:, 0] for x in t], "trust": r if st["trust"] else None,
                               "cand_ap": cand_ap(items, c)}
            print(f"  {name} {lv}: AP {out[(name, lv)]['cand_ap']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return out


def pred_at(preds, m, lv, i):
    p = preds[(m, lv)]
    return {k: (p[k][i] if p[k] is not None else None) for k in ("cand", "pinv", "trust")}


# ---------------------------------------------------------------- assembly and scoring
def cf1_rule(m):
    """Empty-screen rule: a screen with no true and no predicted container scores 1.0."""
    if m["n_true_containers"] == 0 and m["n_pred_containers"] == 0:
        return 1.0
    return m["container_f1"]


def _grid_job(args):
    """Container F1 (raw truth, empty-screen rule) of one screen for a list of (t_c, removed mask)."""
    v, scores, combos = args
    return [cf1_rule(grouping_metrics(v["true"], PH.removal_tree(v, scores, tc, rm))) for tc, rm in combos]


def _rows_job(args):
    v, scores, tc, rm, system, level = args
    pred = PH.removal_tree(v, scores, tc, rm)
    base = {"screen_id": v["screen_id"], "system": system, "level": level, "n_elements": len(v["leaf_ids"]),
            "bin": density_bin(len(v["leaf_ids"])), "n_removed": int(rm.sum()),
            "removed_valid": int((rm & (v["leaf_kind"] == 0)).sum()),
            "kept_unfaithful": int((~rm & (v["leaf_kind"] > 0)).sum())}
    rows = []
    for tname in ("raw", "smoothed"):
        tt = v["true"] if tname == "raw" else smooth_single_child(v["true"])
        m = grouping_metrics(tt, pred)
        rows.append({**base, "truth": tname, **m, "container_f1": cf1_rule(m)})
    return rows


def best_of(grid, values, prefer=0.5):
    """Index of the best mean (rounded to 1e-6); ties: None ("remove nothing") first, then closest to `prefer`."""
    mean = np.asarray(grid, float).mean(0)

    def key(i):
        v = values[i]
        return (round(float(mean[i]), 6), 1 if v is None else 0, -abs((v if v is not None else prefer) - prefer))
    return max(range(len(values)), key=key), mean


def element_trust(source, v, pred, base):
    """p(faithful) per element of one level dict. pred = this model's predictions for the screen."""
    L = len(v["leaf_ids"])
    if source == "trust":
        return pred["trust"][:L]
    if source == "typehead":
        return 1.0 - pred["pinv"]
    sid = v["screen_id"]
    return np.array([base[source].get((sid, int(l)), np.nan) for l in v["leaf_ids"]])


def removal_mask(source, v, pred, base, t_e):
    """Removed elements of one screen (plan §3-4); elements with no score are kept."""
    if source == "oracle":
        return v["leaf_kind"] > 0
    if source == "none" or t_e is None:
        return np.zeros(len(v["leaf_ids"]), bool)
    p = element_trust(source, v, pred, base)
    return np.nan_to_num(p, nan=1.0) < t_e


def tune_system(ex, cfg, vs, prs, source, base):
    """Plan §4 on the validation real level: t_c with nothing removed, then t_e at that t_c, then t_c again."""
    grid = cfg["thresholds"]
    run = lambda jobs: list(ex.map(_grid_job, jobs, chunksize=16))  # noqa: E731
    nothing = [np.zeros(len(v["leaf_ids"]), bool) for v in vs]
    g1 = run([(v, p["cand"], [(t, z) for t in grid]) for v, p, z in zip(vs, prs, nothing)])
    i1, m1 = best_of(g1, grid)
    out = {"t_c_step1": grid[i1], "curve_step1": dict(zip(map(str, grid), m1.round(5).tolist()))}
    if source == "none":
        out.update(t_c=grid[i1], t_e=None)
        return out
    tes = [0.5] if source == "oracle" else [None] + grid
    masks = [[removal_mask(source, v, p, base, te) for te in tes] for v, p in zip(vs, prs)]
    g2 = run([(v, p["cand"], [(grid[i1], m) for m in ms]) for v, p, ms in zip(vs, prs, masks)])
    i2, m2 = best_of(g2, tes)
    out["curve_step2"] = dict(zip(map(str, tes), m2.round(5).tolist()))
    g3 = run([(v, p["cand"], [(t, ms[i2]) for t in grid]) for v, p, ms in zip(vs, prs, masks)])
    i3, m3 = best_of(g3, grid)
    out.update(t_e=None if source == "oracle" else tes[i2], t_c=grid[i3],
               curve_step3=dict(zip(map(str, grid), m3.round(5).tolist())))
    return out


def systems_list(cfg, preds):
    """(system name, model, removal source, levels) of every structure row (plan §5)."""
    models = sorted({m for m, _ in preds})
    out = []
    for m in models:
        out += [(m, m, "none", cfg["levels_eval"]), (f"{m}+typehead", m, "typehead", cfg["levels_eval"]),
                (f"{m}+oracle", m, "oracle", cfg["levels_eval"])]
        if preds[(m, "real")]["trust"] is not None:
            out.append((f"{m}+trust", m, "trust", cfg["levels_eval"]))
    ref = cfg["baseline_model"]            # the earlier models' removal is applied to these candidate scores
    if ref in models:
        out += [(f"{ref}+{b}", ref, b, ["real"]) for b in cfg["baselines"]["names"]]   # CLAY elements only (no fakes)
    return out


def score_split(cfg, split, screens, preds, base, choice=None):
    """Structure rows for every system and level. choice=None: tune on this split (validation only)."""
    ex = ProcessPoolExecutor(cfg["workers"])
    rows, tuned = [], {}
    for sysname, m, src, levels in systems_list(cfg, preds):
        if src not in ("none", "typehead", "oracle", "trust") and src not in base:
            print(f"[{sysname}] no scores on {split}, left out", flush=True)
            continue
        if choice is not None and sysname not in choice:          # never tuned on test
            print(f"[{sysname}] not in the validation choices, left out", flush=True)
            continue
        t0 = time.time()
        if choice is None:
            vs = [s["v"]["real"] for s in screens]
            tuned[sysname] = tune_system(ex, cfg, vs, [pred_at(preds, m, "real", i) for i in range(len(screens))], src, base)
        else:
            tuned[sysname] = choice[sysname]
        tc, te = tuned[sysname]["t_c"], tuned[sysname]["t_e"]
        for lv in levels:
            jobs = []
            for i, s in enumerate(screens):
                v, p = s["v"][lv], pred_at(preds, m, lv, i)
                jobs.append((v, p["cand"], tc, removal_mask(src, v, p, base, te), sysname, lv))
            rows += [x for xx in ex.map(_rows_job, jobs, chunksize=16) for x in xx]
        print(f"[{sysname}] t_c {tc} t_e {te} ({time.time() - t0:.0f}s)", flush=True)
    ex.shutdown()
    return pd.DataFrame(rows), tuned


def summarise_structure(cfg, df):
    summ = {}
    for (lv, tname, sysname), x in df.groupby(["level", "truth", "system"]):
        r = {k: bootstrap_mean(x[k], cfg["n_boot"], cfg["seed"]) for k in METRICS}
        r["container_f1_true_only"] = bootstrap_mean(x.container_f1[x.n_true_containers > 0], cfg["n_boot"], cfg["seed"])
        r["mean_pred_containers"] = float(x.n_pred_containers.mean())
        r["mean_removed"] = float(x.n_removed.mean())
        r["removed_valid"] = int(x.removed_valid.sum())
        r["kept_unfaithful"] = int(x.kept_unfaithful.sum())
        r["by_density_container_f1"] = {b: round(float(v.container_f1.mean()), 4) for b, v in x.groupby("bin")}
        summ[f"{sysname}|{lv}|{tname}"] = r
    paired = {}
    for (lv, tname), x in df.groupby(["level", "truth"]):
        d = x.pivot(index="screen_id", columns="system", values="container_f1")
        for a, b in cfg["pairs"]:
            if a in d and b in d:
                paired[f"{a}_minus_{b}|{lv}|{tname}"] = paired_bootstrap(d[a], d[b], cfg["n_boot"], cfg["seed"])
    seeds = {}
    for suffix in ["", "+trust", "+oracle", "+typehead"]:
        names = [f"fusion_full_s{s}{suffix}" for s in cfg["variants"]["fusion_full"]["seeds"]]
        for (lv, tname), x in df.groupby(["level", "truth"]):
            vals = [x[x.system == n].container_f1.mean() for n in names if (x.system == n).any()]
            if len(vals) > 1:
                seeds[f"fusion_full{suffix}|{lv}|{tname}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                                                           "values": [float(v) for v in vals]}
    return summ, paired, seeds


# ---------------------------------------------------------------- trust metrics (plan §5)
def trust_table(screens, preds, base, level="real", kinds=(0, 1)):
    """Rows = elements of `level` with leaf_kind in kinds; columns = p(faithful) of every source."""
    rows = {"screen_id": [], "y": [], "kind": []}
    cols = {}
    models = sorted({m for m, lv in preds if lv == level})
    for i, s in enumerate(screens):
        v = s["v"][level]
        keep = np.isin(v["leaf_kind"], kinds)
        rows["screen_id"].append(np.full(keep.sum(), s["screen_id"]))
        rows["y"].append(v["leaf_trust"][keep])
        rows["kind"].append(v["leaf_kind"][keep])
        for m in models:
            p = pred_at(preds, m, level, i)
            if p["trust"] is not None:
                cols.setdefault(f"{m}:trust", []).append(element_trust("trust", v, p, base)[keep])
            cols.setdefault(f"{m}:typehead", []).append(element_trust("typehead", v, p, base)[keep])
        for b in base:
            cols.setdefault(b, []).append(element_trust(b, v, None, base)[keep])
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out.update({k: np.concatenate(v).astype(np.float64) for k, v in cols.items()})
    return out


def cand_trust_table(screens, preds, level="real"):
    """Raw candidates with CLAY truth (reported apart from elements)."""
    out = {"screen_id": [], "y": []}
    models = sorted({m for m, lv in preds if lv == level and preds[(m, lv)]["trust"] is not None})
    cols = {m: [] for m in models}
    for i, s in enumerate(screens):
        v = s["v"][level]
        L, t = len(v["leaf_ids"]), v["M"]["trust"]
        keep = t >= 0
        out["screen_id"].append(np.full(keep.sum(), s["screen_id"]))
        out["y"].append(t[keep])
        for m in models:
            cols[m].append(preds[(m, level)]["trust"][i][L:][keep])
    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update({f"{m}:trust": np.concatenate(v).astype(np.float64) for m, v in cols.items()})
    return res


def calibrated(p, y, sid, k, kind, maps, new_maps, seed):
    """Validation (maps None): cross-fitted on screen halves; the map fitted on all validation is kept for test.
    Test: the validation map, never refitted."""
    fit = TR.fit_temperature if kind == "temperature" else TR.fit_platt
    if maps is None:
        new_maps[f"{k}|{kind}"] = fit(p, y)
        return TR.cross_fitted(p, y, sid, fit, seed)
    return TR.apply_cal(maps[f"{k}|{kind}"], p)


def trust_report(cfg, screens, preds, base, maps=None, dev_base_rate=None):
    t = trust_table(screens, preds, base)
    y, sid = t["y"].astype(float), t["screen_id"]
    srcs = [k for k in t if k not in ("screen_id", "y", "kind")]
    keep = np.all([np.isfinite(t[k]) for k in srcs], 0)              # the same elements for every source
    yk, sk = y[keep], sid[keep]
    base_rate = dev_base_rate if dev_base_rate is not None else float(yk.mean())
    nb = cfg["calibration_bins"]
    res = {"n_elements": int(keep.sum()), "n_elements_all": int(len(y)), "n_screens": int(len(np.unique(sk))),
           "coverage": {k: float(np.isfinite(t[k]).mean()) for k in srcs}, "base_rate_faithful_validation": base_rate,
           "sources": {"constant_base_rate": {"raw": TR.trust_scores(np.full(len(yk), base_rate), yk, base_rate, nb)}}}
    new_maps = {}
    for k in srcs:
        p = t[k][keep]
        r = {"raw": TR.trust_scores(p, yk, base_rate, nb), "reliability_raw": TR.reliability(p, yk, nb)}
        for kind in ("temperature", "platt"):
            if maps is not None and f"{k}|{kind}" not in maps:       # no validation scores: raw only
                continue
            pc = calibrated(p, yk, sk, k, kind, maps, new_maps, cfg["seed"])
            r[kind] = TR.trust_scores(pc, yk, base_rate, nb)
            r[f"reliability_{kind}"] = TR.reliability(pc, yk, nb)
        r["auroc_ci"] = bootstrap_rows(sk, TR.auroc_rows(yk, p), cfg["n_boot"], cfg["seed"])
        res["sources"][k] = r
    res["paired_auroc"] = {f"{a}_minus_{b}": bootstrap_rows(sk, TR.auroc_rows(yk, t[a][keep]), cfg["n_boot"], cfg["seed"],
                                                            other=TR.auroc_rows(yk, t[b][keep]))
                           for a, b in cfg["trust_pairs"] if a in t and b in t}
    c = cand_trust_table(screens, preds)
    cy = c["y"].astype(float)
    res["raw_candidates"] = {k: TR.trust_scores(c[k], cy, float(cy.mean()), nb) for k in c if k.endswith(":trust")}
    if "heavy" in cfg["levels_eval"]:
        h = trust_table(screens, preds, {}, level="heavy", kinds=(0, 1, 2))
        res["heavy_level"] = {}
        for k in [k for k in h if k.endswith(":trust")]:
            res["heavy_level"][k] = {}
            for kind, name in ((1, "real_invalid_vs_valid"), (2, "fakes_vs_valid")):
                m = np.isin(h["kind"], (0, kind))
                yy = h["y"][m].astype(float)
                res["heavy_level"][k][name] = TR.trust_scores(h[k][m], yy, float(yy.mean()), nb)
    return res, new_maps


# ---------------------------------------------------------------- dev / test stages
def report(cfg, split, run_dir, choice=None):
    screens = load_cache(cfg, split)
    t0 = time.time()
    preds = predict_split(cfg, split, screens, cfg["levels_eval"])
    pickle.dump(preds, open(out_dir(cfg) / f"preds_{split}{sfx(cfg)}.pkl", "wb"))
    print(f"predictions done ({time.time() - t0:.0f}s)", flush=True)
    base = baseline_scores(cfg, split)
    df, tuned = score_split(cfg, split, screens, preds, base, None if choice is None else choice["systems"])
    df.to_csv(run_dir / "per_screen.csv.gz", index=False)
    summ, paired, seeds = summarise_structure(cfg, df)
    tr, maps = trust_report(cfg, screens, preds, base, None if choice is None else choice["calibration"],
                            None if choice is None else choice["base_rate"])
    res = {"split": split, "n_screens": len(screens),
           "cand_ap": {f"{m}|{lv}": p["cand_ap"] for (m, lv), p in preds.items()},
           "tuned": tuned, "structure": summ, "paired": paired, "seeds": seeds, "trust": tr,
           "notes": ["container_f1: empty-screen rule (both empty = 1.0); container_f1_true_only = screens with >= 1 true container",
                     "earlier models' removal: real level only (their scores exist for CLAY elements, not injected fakes)",
                     "trust: real-level elements (valid leaves + CLAY invalid elements), unfaithful = positive; same "
                     "elements for every source"]}
    if choice is None:
        _json(out_dir(cfg) / f"dev_choice{sfx(cfg)}.json",
              {"systems": tuned, "calibration": maps, "base_rate": tr["base_rate_faithful_validation"],
               "models": sorted({m for m, _ in preds})})
    return res


def stage_dev(cfg, a, run_dir):
    return report(cfg, "dev", run_dir)


def stage_test(cfg, a, run_dir):
    marker = out_dir(cfg) / f"TEST_DONE{sfx(cfg)}"
    assert not marker.exists(), "the test split was already scored once; it is not run again"
    choice = _json(out_dir(cfg) / f"dev_choice{sfx(cfg)}.json")
    assert sorted(trained(cfg)) == choice["models"], "trained models differ from the validation report"
    res = report(cfg, "test", run_dir, choice)
    marker.write_text("scored once\n")
    return res


def print_summary(res):
    """Compact console summary (Colab output is cut at about 6,000 characters)."""
    if "history" in res:
        print(f"{res['name']}: dev AP {res['dev_ap']:.4f}, best epoch {res['best_epoch']}, {res['minutes']} min")
        return
    if "structure" not in res:
        print({k: v for k, v in res.items() if not isinstance(v, (list, dict))})
        return
    print(f"{'system (real level, raw truth)':36s} {'contF1':>7s}  t_c / t_e")
    for k, v in sorted(res["structure"].items()):
        if k.endswith("|real|raw"):
            s = k.split("|")[0]
            tu = res["tuned"].get(s, {})
            print(f"{s:36s} {v['container_f1']['mean']:7.3f}  {tu.get('t_c')} / {tu.get('t_e')}")
    print("by level (raw truth):")
    for k, v in sorted(res["structure"].items()):
        s, lv, tn = k.split("|")
        if tn == "raw" and s in ("fusion_full_s0", "fusion_full_s0+trust", "fusion_full_s0+oracle"):
            print(f"  {s:22s} {lv:8s} {v['container_f1']['mean']:.3f}")
    for k, v in res["paired"].items():
        if "|real|raw" in k:
            print(f"{k[:-9]:52s} {v['diff']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] p={v['p_a_not_better']:.3f}")
    tr = res["trust"]
    print(f"trust, {tr['n_elements']} elements: AUROC / ECE raw / ECE temperature")
    for k, v in tr["sources"].items():
        if "reliability_raw" in v:
            te = f"{v['temperature']['ece']:.3f}" if "temperature" in v else "  -  "
            print(f"  {k:26s} {v['raw']['auroc']:.3f} / {v['raw']['ece']:.3f} / {te}")
    for k, v in tr["paired_auroc"].items():
        print(f"  {k:44s} {v['value']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] p={v['p_a_not_better']:.3f}")


# ---------------------------------------------------------------- main
STAGES = {"train": stage_train, "dev": stage_dev, "test": stage_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["rates", "fingerprint", "baselines", "build", "embed", "train", "dev", "test"])
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--laptop", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.laptop, a.smoke)
    if a.stage == "test":                     # checked before a run folder is made
        assert a.confirm_test, "the test stage runs once: pass --confirm-test"
        assert not (Path(cfg["out_dir"]) / f"TEST_DONE{sfx(cfg)}").exists(), "the test split was already scored once"
    tag = {"build": f"_{a.split}", "embed": f"_{a.split}", "train": f"_{a.variant}_s{a.seed}"}.get(a.stage, "")
    run_dir = start_run(f"trust_score_{a.stage}{tag}{sfx(cfg)}", {**cfg, "stage": a.stage, "split": a.split,
                                                        "variant": a.variant, "seed_arg": a.seed}, root=cfg["runs_root"])
    fn = {"rates": lambda: stage_rates(cfg), "fingerprint": lambda: stage_fingerprint(cfg),
          "baselines": lambda: stage_baselines(cfg), "build": lambda: stage_build(cfg, a.split),
          "embed": lambda: stage_embed(cfg, a.split), **{k: (lambda k=k: STAGES[k](cfg, a, run_dir)) for k in STAGES}}
    res = fn[a.stage]()
    finish_run(run_dir, res)
    print_summary(res)
    print("run dir:", run_dir)


if __name__ == "__main__":
    main()
