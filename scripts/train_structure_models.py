"""Learned candidate-container nesting head for Models 4-6, plus its comparators on the same candidates.
Config: configs/train_structure_models.yaml (Colab A100, notebooks/train_structure_models_pixels_metadata_fusion.ipynb).

Stages
  build  --split train|dev|test  leaves + candidate pools + labels + rule-baseline trees -> cache.
                                 Prints an integer fingerprint so a laptop build can be checked against Colab's.
  embed  --split ...             crop embeddings of every leaf / candidate box with each backbone (data/nesting_crops.py)
  tune   --model model4|5|6      random search, equal budget for every model; trial 0 = config settings; models with
                                 crops first run trial 0 with each backbone and keep the better one (rule in config).
                                 Selection: validation candidate average precision. Resumable (trials saved).
  gbdt   --model gbdt_*          the trees comparator, same budget and selection rule
  dev    validation report: selected models, untuned extra rows, assembler thresholds tuned on validation container F1,
         pool ceilings, rule baselines, paired bootstrap on screens. Writes the choices the test stage uses.
  test   ONCE (--confirm-test): the dev stage's choices applied to the clean test split; leaf predictions saved
--smoke  laptop check on the first N screens (config `smoke` block; run names get _smoke; not results)
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
from uistruct.data.features import FLAGS
from uistruct.data.nesting_tokens import build_screen, rule_trees
from uistruct.eval.metrics import grouping_metrics
from uistruct.eval.stats import bootstrap_mean, paired_bootstrap
from uistruct.runlog import finish_run, start_run
from uistruct.train.nesting_head import (TokenEncoder, build_model, cand_ap, fit_gbdt, gbdt_predict, predict,
                                         screen_tree, train_scorer, tree_rows, tune_threshold)
from uistruct.train.type_head import per_screen_confusion, scores as type_scores

BOX = ["nx0", "ny0", "nx1", "ny1"]
METRICS = ["container_f1", "container_match", "container_precision", "edge_f1", "leaf_edge_f1", "edit_distance"]
RAW_COLS = ["screen_id", "node_idx", "parent_idx", "box_ok", *BOX, "depth", "n_children", "class_name", *FLAGS]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROP_DIM = 512
CEILING_NOTE = ("ceiling_*|smoothed = raw-truth oracle picks scored on smoothed truth (like-for-like with models "
                "trained on raw truth); not the candidate-pool check's smoothed ceiling, which picks on smoothed truth")


# ---------------------------------------------------------------- config and files
def load_cfg(path, smoke):
    cfg = yaml.safe_load(open(path))
    if smoke:
        for k, v in cfg["smoke"].items():
            cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    cfg["is_smoke"] = bool(smoke)
    cfg["workers"] = cfg.get("workers") or os.cpu_count()
    return cfg


def sfx(cfg):
    return "_smoke" if cfg["is_smoke"] else ""


def cache_path(cfg, split):
    return Path(cfg["cache_dir"]) / f"screens_{split}{sfx(cfg)}.pkl"


def load_split(cfg, split):
    with open(cache_path(cfg, split), "rb") as f:
        return pickle.load(f)


def trial_dir(cfg, name, tag=None):
    p = Path(cfg["out_dir"]) / "trials" / name
    return p / tag if tag else p


def fingerprint(screens):
    """sha256 over integers only (screen ids, leaf ids, candidate membership, labels): equal on any machine."""
    h = hashlib.sha256()
    for s in screens:
        h.update(np.int64(s["screen_id"]).tobytes())
        h.update(np.asarray(s["leaf_ids"], np.int64).tobytes())
        for k in ("P", "M"):
            h.update(np.packbits(s[k]["member"]).tobytes())
            h.update(s[k]["y"].astype(np.int8).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- build
def _build_one(args):
    g, raw, cfg = args
    sc = build_screen(g, raw, cfg["pool"], cfg["min_iou"])
    sc["rules"] = rule_trees(g, raw, cfg["xy_tol"])
    return sc


def stage_build(cfg, split):
    t0 = time.time()
    el = pd.read_parquet(f"{cfg['data_dir']}/elements.parquet",
                         columns=["screen_id", "node_idx", "gt_parent", "label", "is_valid", "split_clean", *BOX])
    el = el[el.is_valid & (el.split_clean == split)]
    sids = sorted(el.screen_id.unique())
    if cfg.get("limit"):
        sids = sids[:cfg["limit"]]
        el = el[el.screen_id.isin(sids)]
    nodes = pd.read_parquet(f"{cfg['data_dir']}/nodes.parquet", columns=RAW_COLS)
    nodes = nodes[nodes.screen_id.isin(set(sids))]
    raw_by = {k: v.sort_values("node_idx") for k, v in nodes.groupby("screen_id", sort=False)}
    del nodes
    groups = dict(tuple(el.groupby("screen_id", sort=True)))
    del el
    out = []
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        for w in range(0, len(sids), 2048):                       # bounded windows (memory)
            win = sids[w:w + 2048]
            out.extend(ex.map(_build_one, [(groups[s], raw_by[s], cfg) for s in win], chunksize=32))
            print(f"  {min(w + 2048, len(sids))}/{len(sids)} screens ({time.time() - t0:.0f}s)", flush=True)
    path = cache_path(cfg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {"split": split, "n_screens": len(out), "sec": round(time.time() - t0, 1), "fingerprint": fingerprint(out)}
    for k in ("P", "M"):
        summary[f"{k}_mean_candidates"] = float(np.mean([len(s[k]["sets"]) for s in out]))
        summary[f"{k}_mean_positives"] = float(np.mean([s[k]["y"].sum() for s in out]))
        summary[f"{k}_max_candidates"] = int(max(len(s[k]["sets"]) for s in out))
    summary["mean_leaves"] = float(np.mean([len(s["leaf_ids"]) for s in out]))
    return summary


# ---------------------------------------------------------------- embed
def backbones(cfg):
    from uistruct.models.cnn import resnet18_trunk
    out = {}
    for name in cfg["backbones"]:
        if name == "model3":            # Model 3's fine-tuned trunk, frozen here
            st = torch.load(cfg["model3_ckpt"], map_location="cpu", weights_only=False)
            assert st["fine_tune"] and abs(st["hp"]["lr_backbone"] - cfg["model3_lr_backbone"]) < 1e-12, \
                "checkpoint is not the selected Model 3"
            m = resnet18_trunk(pretrained=False)
            m.load_state_dict({k[len("backbone."):]: v for k, v in st["state"].items() if k.startswith("backbone.")})
        elif name == "imagenet":
            m = resnet18_trunk(pretrained=cfg.get("pretrained", True))
        else:
            raise ValueError(name)
        out[name] = m
    return out


def stage_embed(cfg, split):
    assert cfg["crop_size"] == NC.SIZE
    t0 = time.time()
    screens = load_split(cfg, split)
    out, index = NC.embed_screens(screens, cfg["img_dir"], backbones(cfg), DEVICE, cfg["workers"])
    os.makedirs(cfg["emb_dir"], exist_ok=True)
    NC.save(out, index, cfg["emb_dir"], split)
    n = len(next(iter(out.values())))
    return {"split": split, "n_screens": len(screens), "n_boxes": n, "boxes_per_screen": n / len(screens),
            "backbones": list(out), "sec": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- inputs per model
def split_crops(cfg, split, screens, backbone, pool):
    e, idx = NC.load(cfg["emb_dir"], split, backbone)
    e = np.asarray(e)                                             # into RAM: fast fancy indexing
    return [NC.token_crops(e, idx[s["screen_id"]], pool) for s in screens]


def prepare(cfg, spec, backbone, tr, others):
    """Encoder fitted on train only; transformed train items and transformed items of the other splits."""
    pool, use_meta = spec["pool"], spec["use_meta"]
    enc = TokenEncoder(pool, use_meta).fit(tr)
    if spec["use_crop"]:
        ctr = split_crops(cfg, "train", tr, backbone, pool)
        enc.fit_crops([c[0] for c in ctr], [c[1] for c in ctr])
        trx = [enc.transform(s, c) for s, c in zip(tr, ctr)]
        del ctr
        ox = {sp: [enc.transform(s, c) for s, c in zip(scr, split_crops(cfg, sp, scr, backbone, pool))]
              for sp, scr in others.items()}
    else:
        trx = [enc.transform(s) for s in tr]
        ox = {sp: [enc.transform(s) for s in scr] for sp, scr in others.items()}
    return enc, trx, ox


def draw(space, rng):
    hp = {}
    for k, v in space.items():
        hp[k] = float(np.exp(rng.uniform(np.log(v["log"][0]), np.log(v["log"][1])))) if isinstance(v, dict) \
            else v[int(rng.integers(len(v)))]
    return hp


def trial_settings(base, tune):
    """Trial 0 = the config settings; trials 1.. drawn once from a fixed seed, so every model of a kind
    gets exactly the same candidate settings (like-for-like, as for Model 2)."""
    rng = np.random.default_rng(tune["seed"])
    out = [dict(base)]
    for _ in range(tune["trials"] - 1):
        hp = {**base, **draw(tune["space"], rng)}
        if "d" in hp:
            hp["ffn"] = 2 * hp["d"]
        out.append(hp)
    return out


def _json(p, obj=None):
    if obj is None:
        return json.load(open(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(p, "w"), indent=1, default=float)
    return obj


# ---------------------------------------------------------------- neural scorers
def fit_neural(cfg, name, tag, spec, hp, backbone, tr, dv, prepared, seed=None):
    d = trial_dir(cfg, name, tag)
    if (d / "trial.json").exists():
        print(f"[{name}/{tag}] already saved, skipped", flush=True)
        return _json(d / "trial.json")
    if backbone not in prepared:
        prepared.clear()                                          # one prepared input set in RAM at a time
        prepared[backbone] = prepare(cfg, spec, backbone, tr, {"dev": dv})
    enc, trx, ox = prepared[backbone]
    t0 = time.time()
    print(f"[{name}/{tag}] training: {json.dumps(hp)}", flush=True)
    model, hist = train_scorer(hp, enc, trx, ox["dev"], cfg["seed"] if seed is None else seed,
                               CROP_DIM if spec["use_crop"] else 0,
                               device=DEVICE, workers=cfg["loader_workers"])
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "hp": hp, "spec": spec,
                "backbone": backbone}, d / "model.pt")
    pickle.dump(enc, open(d / "enc.pkl", "wb"))
    best = int(np.argmax([h["dev_ap"] for h in hist]))
    return _json(d / "trial.json", {"name": name, "tag": tag, "hp": hp, "backbone": backbone, "history": hist,
                                    "seed": cfg["seed"] if seed is None else seed,
                                    "dev_ap": hist[best]["dev_ap"], "best_epoch": best + 1,
                                    "minutes": round((time.time() - t0) / 60, 1)})


def pick_backbone(cfg, r):
    """Rule fixed before any result: higher validation AP; within `backbone_tie` -> imagenet (its features
    never saw training labels)."""
    if len(r) == 1:
        return next(iter(r))
    a, b = r["model3"]["dev_ap"], r["imagenet"]["dev_ap"]
    return "imagenet" if abs(a - b) <= cfg["backbone_tie"] or b > a else "model3"


def stage_tune(cfg, name):
    spec = cfg["models"][name]
    tr, dv = load_split(cfg, "train"), load_split(cfg, "dev")
    sets = trial_settings(cfg["scorer"], cfg["tune"])
    prepared = {}
    if spec["use_crop"]:
        r0 = {b: fit_neural(cfg, name, f"t0_{b}", spec, sets[0], b, tr, dv, prepared) for b in cfg["backbones"]}
        bb = pick_backbone(cfg, r0)
        trials = [r0[bb]] + [fit_neural(cfg, name, f"t{k}_{bb}", spec, sets[k], bb, tr, dv, prepared)
                             for k in range(1, len(sets))]
        extra = {"backbone_trial0_ap": {b: v["dev_ap"] for b, v in r0.items()}, "backbone": bb}
    else:
        trials = [fit_neural(cfg, name, f"t{k}", spec, sets[k], None, tr, dv, prepared) for k in range(len(sets))]
        extra = {"backbone": None}
    best = max(trials, key=lambda t: t["dev_ap"])                 # ties -> earliest trial
    _json(trial_dir(cfg, name) / "best.json", {"tag": best["tag"], "hp": best["hp"], "backbone": best["backbone"],
                                               "dev_ap": best["dev_ap"]})
    return {"model": name, **extra, "best": best["tag"], "best_dev_ap": best["dev_ap"],
            "trials": {t["tag"]: {k: t[k] for k in ("hp", "dev_ap", "best_epoch", "minutes")} for t in trials}}


def load_neural(cfg, name, tag):
    d = trial_dir(cfg, name, tag)
    st = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    enc = pickle.load(open(d / "enc.pkl", "rb"))
    model = build_model(st["hp"], enc, CROP_DIM if st["spec"]["use_crop"] else 0)
    model.load_state_dict(st["state"])
    return model.to(DEVICE), enc, st


def neural_probs(cfg, name, tag, split, screens):
    model, enc, st = load_neural(cfg, name, tag)
    spec = st["spec"]
    if spec["use_crop"]:
        items = [enc.transform(s, c) for s, c in zip(screens, split_crops(cfg, split, screens, st["backbone"], spec["pool"]))]
    else:
        items = [enc.transform(s) for s in screens]
    probs, tprobs = predict(model, items, st["hp"]["budget"], DEVICE, cfg["loader_workers"])
    return probs, tprobs, cand_ap(items, probs), spec


# ---------------------------------------------------------------- trees comparator
def fit_tree(cfg, name, tag, spec, hp, backbone, tr, dv):
    d = trial_dir(cfg, name, tag)
    if (d / "trial.json").exists():
        print(f"[{name}/{tag}] already saved, skipped", flush=True)
        return _json(d / "trial.json")
    t0 = time.time()
    pool, use_meta = spec["pool"], spec["use_meta"]
    enc = TokenEncoder(pool, use_meta).fit(tr)
    ctr = cdv = None
    if spec["use_crop"]:
        ctr = split_crops(cfg, "train", tr, backbone, pool)
        cdv = split_crops(cfg, "dev", dv, backbone, pool)
        enc.fit_crops([c[0] for c in ctr], [c[1] for c in ctr])
    print(f"[{name}/{tag}] fitting: {json.dumps(hp)}", flush=True)
    clf = fit_gbdt(hp, tr, pool, use_meta, enc, cfg["seed"], ctr, cfg["pca_dim"])
    gp = gbdt_predict(clf, dv, pool, use_meta, enc, cdv)
    d.mkdir(parents=True, exist_ok=True)
    pickle.dump({"clf": clf, "enc": enc, "spec": spec, "backbone": backbone, "hp": hp}, open(d / "clf.pkl", "wb"))
    ap = cand_ap([{"y": s[pool]["y"]} for s in dv], gp)
    return _json(d / "trial.json", {"name": name, "tag": tag, "hp": hp, "backbone": backbone, "dev_ap": ap,
                                    "n_iter": int(clf.n_iter_), "minutes": round((time.time() - t0) / 60, 1)})


def tree_backbone(cfg, spec):
    return _json(trial_dir(cfg, spec["pair"]) / "best.json")["backbone"] if spec["use_crop"] else None


def stage_gbdt(cfg, name):
    spec = cfg["gbdt_models"][name]
    bb = tree_backbone(cfg, spec)
    tr, dv = load_split(cfg, "train"), load_split(cfg, "dev")
    sets = trial_settings(cfg["gbdt"], cfg["gbdt_tune"])
    trials = [fit_tree(cfg, name, f"t{k}", spec, sets[k], bb, tr, dv) for k in range(len(sets))]
    best = max(trials, key=lambda t: t["dev_ap"])
    _json(trial_dir(cfg, name) / "best.json", {"tag": best["tag"], "hp": best["hp"], "backbone": bb,
                                               "dev_ap": best["dev_ap"]})
    return {"model": name, "backbone": bb, "best": best["tag"], "best_dev_ap": best["dev_ap"],
            "trials": {t["tag"]: {k: t[k] for k in ("hp", "dev_ap", "n_iter", "minutes")} for t in trials}}


def tree_probs(cfg, name, tag, split, screens):
    o = pickle.load(open(trial_dir(cfg, name, tag) / "clf.pkl", "rb"))
    spec = o["spec"]
    crops = split_crops(cfg, split, screens, o["backbone"], spec["pool"]) if spec["use_crop"] else None
    gp = gbdt_predict(o["clf"], screens, spec["pool"], spec["use_meta"], o["enc"], crops)
    return gp, cand_ap([{"y": s[spec["pool"]]["y"]} for s in screens], gp), spec


# ---------------------------------------------------------------- extra (untuned) rows
def ensure_extras(cfg, tr, dv):
    """Ablation rows trained once with the settings of the tuned model they are compared with."""
    for name, spec in cfg.get("extra_models", {}).items():
        src = _json(trial_dir(cfg, spec["settings_from"]) / "best.json")
        bb = src["backbone"] if spec["use_crop"] else None
        fit_neural(cfg, name, "final", spec, src["hp"], bb, tr, dv, {})
        _json(trial_dir(cfg, name) / "best.json", {"tag": "final", "hp": src["hp"], "backbone": bb,
                                                   "settings_from": spec["settings_from"]})
    for name, spec in cfg.get("gbdt_extra", {}).items():
        src = _json(trial_dir(cfg, spec["settings_from"]) / "best.json")
        bb = tree_backbone(cfg, spec)
        fit_tree(cfg, name, "final", spec, src["hp"], bb, tr, dv)
        _json(trial_dir(cfg, name) / "best.json", {"tag": "final", "hp": src["hp"], "backbone": bb,
                                                   "settings_from": spec["settings_from"]})


def systems(cfg):
    """{name: kind} of every learned system in the reports."""
    out = {n: "neural" for n in cfg["models"]}
    out.update({n: "neural" for n in cfg.get("extra_models", {})})
    out.update({n: "tree" for n in cfg["gbdt_models"]})
    out.update({n: "tree" for n in cfg.get("gbdt_extra", {})})
    return out


# ---------------------------------------------------------------- scoring
def _rows_job(args):
    sc, pool, probs, thr, system = args
    return tree_rows(sc, screen_tree(sc, pool, probs, thr), system)


def _rule_job(args):
    sc, name = args
    return tree_rows(sc, sc["rules"][name], name)


def score_split(cfg, split, screens, thresholds=None):
    """Every system on one split. thresholds=None: tune on this split (validation only)."""
    ex = ProcessPoolExecutor(cfg["workers"])
    mapper = lambda f, it: ex.map(f, it, chunksize=16)  # noqa: E731
    rows, out, leaf_pred = [], {}, {}
    for name, kind in systems(cfg).items():
        best = _json(trial_dir(cfg, name) / "best.json")
        t0 = time.time()
        if kind == "neural":
            probs, tprobs, ap, spec = neural_probs(cfg, name, best["tag"], split, screens)
        else:
            (probs, ap, spec), tprobs = tree_probs(cfg, name, best["tag"], split, screens), None
        pool = spec["pool"]
        r = {"pool": pool, "use_meta": spec["use_meta"], "use_crop": spec["use_crop"], "trial": best["tag"],
             "backbone": best["backbone"], "hp": best["hp"], f"{split}_cand_ap": ap}
        if thresholds is None:
            r["threshold"], r["threshold_curve"] = tune_threshold(screens, pool, probs, cfg["thresholds"], mapper)
        else:
            r["threshold"] = thresholds[name]
        rows += [x for xx in mapper(_rows_job, [(s, pool, p, r["threshold"], name) for s, p in zip(screens, probs)])
                 for x in xx]
        if tprobs is not None:
            pred = np.concatenate(tprobs).argmax(1)
            cm = per_screen_confusion(np.concatenate([[s["screen_id"]] * len(s["leaf_ids"]) for s in screens]),
                                      np.concatenate([s["leaf_label"] for s in screens]), pred).sum(0)
            r["leaf_type"] = type_scores(cm)
            leaf_pred[name] = np.concatenate(tprobs).astype(np.float16)
        r["sec"] = round(time.time() - t0, 1)
        out[name] = r
        print(f"[{name}] {split} AP {ap:.4f}, threshold {r['threshold']} ({r['sec']:.0f}s)", flush=True)
    for pool in sorted({s["pool"] for s in cfg["models"].values()}):
        rows += [x for xx in mapper(_rows_job, [(s, pool, s[pool]["y_iou"], cfg["min_iou"], f"ceiling_{pool}")
                                                for s in screens]) for x in xx]
    for rule in ["xycut", "raw_chain_collapsed"]:
        rows += [x for xx in mapper(_rule_job, [(s, rule) for s in screens]) for x in xx]
    ex.shutdown()
    return pd.DataFrame(rows), out, leaf_pred


def summarise(cfg, df):
    summ = {}
    for (tname, sysname), x in df.groupby(["truth", "system"]):
        r = {k: bootstrap_mean(x[k], cfg["n_boot"], cfg["seed"]) for k in METRICS}
        r["mean_pred_containers"] = float(x.n_pred_containers.mean())
        r["mean_true_containers"] = float(x.n_true_containers.mean())
        r["by_density_container_f1"] = {b: round(float(v.container_f1.mean()), 4) for b, v in x.groupby("bin")}
        summ[f"{sysname}|{tname}"] = r
    paired = {}
    for tname in ["raw", "smoothed"]:
        d = df[df.truth == tname].pivot(index="screen_id", columns="system", values="container_f1")
        for a, b in cfg["pairs"]:
            paired[f"{a}_minus_{b}|{tname}"] = paired_bootstrap(d[a], d[b], cfg["n_boot"], cfg["seed"])
    return summ, paired


def stage_dev(cfg, run_dir):
    tr, dv = load_split(cfg, "train"), load_split(cfg, "dev")
    ensure_extras(cfg, tr, dv)
    del tr
    df, out, _ = score_split(cfg, "dev", dv)
    df.to_csv(run_dir / "per_screen.csv", index=False)
    summ, paired = summarise(cfg, df)
    choice = {"thresholds": {n: v["threshold"] for n, v in out.items()},
              "trials": {n: v["trial"] for n, v in out.items()}}
    _json(Path(cfg["out_dir"]) / f"dev_choice{sfx(cfg)}.json", choice)
    return {"n_dev_screens": len(dv), "learned": out, "systems": summ, "paired": paired, "choice": choice,
            "notes": [CEILING_NOTE]}


def stage_test(cfg, run_dir):
    marker = Path(cfg["out_dir"]) / f"TEST_DONE{sfx(cfg)}"
    assert not marker.exists(), "the test split was already scored once; it is not run again"
    choice = _json(Path(cfg["out_dir"]) / f"dev_choice{sfx(cfg)}.json")
    for name in systems(cfg):                                     # the same selected trials as the dev report
        assert _json(trial_dir(cfg, name) / "best.json")["tag"] == choice["trials"][name], name
    te = load_split(cfg, "test")
    df, out, leaf_pred = score_split(cfg, "test", te, choice["thresholds"])
    df.to_csv(run_dir / "per_screen.csv", index=False)
    summ, paired = summarise(cfg, df)
    np.savez_compressed(Path(cfg["out_dir"]) / f"structure_test_leaf_probs{sfx(cfg)}.npz",
                        screen_id=np.concatenate([[s["screen_id"]] * len(s["leaf_ids"]) for s in te]),
                        node_idx=np.concatenate([s["leaf_ids"] for s in te]),
                        label=np.concatenate([s["leaf_label"] for s in te]), **leaf_pred)
    marker.write_text("scored once\n")
    return {"n_test_screens": len(te), "learned": out, "systems": summ, "paired": paired, "notes": [CEILING_NOTE]}


# ---------------------------------------------------------------- extra checks, run in the provenance Colab session
#   seeds --model m    Models 4-6: seeds 1 and 2 at the selected seed-0 settings (no re-tuning)
#   gbdt_refit         gbdt_P_crop re-fit at its selected settings with a 3,000-round cap; validation AP decides
#   leftover_dev       validation: thresholds for the new fits (same rule as seed 0), every Models 4-6 system re-tuned under
#                      the empty-screen container F1 rule (sensitivity only), seed means, paired tests
#   leftover_test      ONCE (--confirm-test): the leftover_dev choices on test; headline thresholds stay as fixed
def stage_seeds(cfg, name):
    spec, best = cfg["models"][name], _json(trial_dir(cfg, name) / "best.json")
    tr, dv = load_split(cfg, "train"), load_split(cfg, "dev")
    prepared, out = {}, {}
    for s in cfg["leftovers"]["seeds"]:
        r = fit_neural(cfg, name, f"{best['tag']}_s{s}", spec, best["hp"], best["backbone"], tr, dv, prepared, seed=s)
        out[f"s{s}"] = {k: r[k] for k in ("dev_ap", "best_epoch", "minutes")}
    return {"model": name, "seed0_tag": best["tag"], "seed0_dev_ap": best["dev_ap"], "seeds": out}


def stage_gbdt_refit(cfg):
    name = cfg["leftovers"]["refit_model"]
    spec, best = cfg["gbdt_models"][name], _json(trial_dir(cfg, name) / "best.json")
    tr, dv = load_split(cfg, "train"), load_split(cfg, "dev")
    hp = {**best["hp"], "max_iter": cfg["leftovers"]["refit_max_iter"]}
    r = fit_tree(cfg, name, "refit", spec, hp, best["backbone"], tr, dv)
    chosen = "refit" if r["dev_ap"] > best["dev_ap"] else best["tag"]       # tie -> the original fit
    _json(Path(cfg["out_dir"]) / f"refit_choice{sfx(cfg)}.json", {"chosen": chosen})
    return {"model": name, "original": {"tag": best["tag"], "dev_ap": best["dev_ap"]},
            "refit": {k: r[k] for k in ("dev_ap", "n_iter", "minutes")}, "chosen": chosen}


def cf1_rule_df(df):
    """Empty-screen rule on per-screen rows: no true and no predicted container = 1.0."""
    both = df.container_f1.isna() & (df.n_true_containers == 0) & (df.n_pred_containers == 0)
    df.loc[both, "container_f1"] = 1.0
    return df


def _f1_rule_at(args):
    sc, pool, scores, thresholds = args
    out = []
    for t in thresholds:
        m = grouping_metrics(sc["true"], screen_tree(sc, pool, scores, t))
        out.append(1.0 if m["n_true_containers"] == 0 and m["n_pred_containers"] == 0 else m["container_f1"])
    return out


def tune_rule(screens, pool, probs, thresholds, mapper):
    """Threshold under the empty-screen rule (every screen counts); ties -> closest to 0.5."""
    f1 = np.array(list(mapper(_f1_rule_at, [(s, pool, p, thresholds) for s, p in zip(screens, probs)])))
    mean = f1.mean(0)
    best = max(range(len(thresholds)), key=lambda i: (round(mean[i], 6), -abs(thresholds[i] - 0.5)))
    return float(thresholds[best]), {f"{t:.2f}": float(m) for t, m in zip(thresholds, mean)}


def leftover_entries(cfg):
    """(row name, kind, model, tag, role) of every system the leftover report scores."""
    kinds = systems(cfg)
    best = {n: _json(trial_dir(cfg, n) / "best.json")["tag"] for n in kinds}
    out = [(n, k, n, best[n], "seed0") for n, k in kinds.items()]
    for n in cfg["models"]:
        out += [(f"{n}_s{s}", "neural", n, f"{best[n]}_s{s}", "seed") for s in cfg["leftovers"]["seeds"]]
    rm = cfg["leftovers"]["refit_model"]
    out.append((f"{rm}_refit", "tree", rm, "refit", "refit"))
    return out


def leftover_report(cfg, split, run_dir, choice=None):
    screens = load_split(cfg, split)
    fixed_thr = _json(Path(cfg["out_dir"]) / f"dev_choice{sfx(cfg)}.json")["thresholds"]          # fixed before the test run
    ex = ProcessPoolExecutor(cfg["workers"])
    mapper = lambda f, it: ex.map(f, it, chunksize=16)  # noqa: E731
    rows, info, new = [], {}, {"seed_thresholds": {}, "retuned": {}}
    for row, kind, name, tag, role in leftover_entries(cfg):
        t0 = time.time()
        if kind == "neural":
            probs, _, ap, spec = neural_probs(cfg, name, tag, split, screens)
        else:
            probs, ap, spec = tree_probs(cfg, name, tag, split, screens)
        pool = spec["pool"]
        if role == "seed0":
            thr = fixed_thr[name]
        elif choice is None:                     # new fits: the same threshold rule seed 0 had
            thr, _ = tune_threshold(screens, pool, probs, cfg["thresholds"], mapper)
        else:
            thr = choice["seed_thresholds"][row]
        if role != "seed0":
            new["seed_thresholds"][row] = thr
        rows += [x for xx in mapper(_rows_job, [(s, pool, p, thr, row) for s, p in zip(screens, probs)]) for x in xx]
        info[row] = {f"{split}_cand_ap": ap, "threshold": thr, "pool": pool}
        if role == "seed0":                      # sensitivity: threshold re-tuned on validation under the empty-screen rule
            rt = tune_rule(screens, pool, probs, cfg["thresholds"], mapper)[0] if choice is None else choice["retuned"][row]
            new["retuned"][row] = rt
            rows += [x for xx in mapper(_rows_job, [(s, pool, p, rt, f"{row}@retuned") for s, p in zip(screens, probs)])
                     for x in xx]
            info[row]["threshold_retuned"] = rt
        print(f"[{row}] {split} AP {ap:.4f} thr {thr} ({time.time() - t0:.0f}s)", flush=True)
    for rule in ["xycut", "raw_chain_collapsed"]:
        rows += [x for xx in mapper(_rule_job, [(s, rule) for s in screens]) for x in xx]
    ex.shutdown()
    df = cf1_rule_df(pd.DataFrame(rows))
    summ = {}
    for (tname, sysname), x in df.groupby(["truth", "system"]):
        summ[f"{sysname}|{tname}"] = {"container_f1": bootstrap_mean(x.container_f1, cfg["n_boot"], cfg["seed"]),
                                      "container_f1_true_only": bootstrap_mean(x.container_f1[x.n_true_containers > 0],
                                                                               cfg["n_boot"], cfg["seed"])}
    seeds, paired = {}, {}
    for tname in ("raw", "smoothed"):
        d = df[df.truth == tname].pivot(index="screen_id", columns="system", values="container_f1")
        for n in cfg["models"]:
            cols = [n] + [f"{n}_s{s}" for s in cfg["leftovers"]["seeds"]]
            v = [float(d[c].mean()) for c in cols]
            seeds[f"{n}|{tname}"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v}
            d[f"{n}_seedmean"] = d[cols].mean(1)            # per-screen mean over the 3 seeds
        for a, b in cfg["leftovers"]["pairs"]:
            if a in d and b in d:
                paired[f"{a}_minus_{b}|{tname}"] = paired_bootstrap(d[a], d[b], cfg["n_boot"], cfg["seed"])
    df.to_csv(run_dir / "per_screen.csv.gz", index=False)
    return {"split": split, "n_screens": len(screens), "systems": info, "summary": summ, "seeds": seeds,
            "paired": paired, "notes": ["container_f1: empty-screen rule; @retuned rows = sensitivity only (headline "
                                        "thresholds were fixed before the test run)"]}, new


def stage_leftover_dev(cfg, run_dir):
    res, new = leftover_report(cfg, "dev", run_dir)
    new["refit_choice"] = _json(Path(cfg["out_dir"]) / "refit_choice.json")["chosen"] if \
        (Path(cfg["out_dir"]) / "refit_choice.json").exists() else None
    _json(Path(cfg["out_dir"]) / f"leftover_choice{sfx(cfg)}.json", new)
    return res


def stage_leftover_test(cfg, run_dir):
    marker = Path(cfg["out_dir"]) / f"LEFTOVER_TEST_DONE{sfx(cfg)}"
    assert not marker.exists(), "the leftover test was already run once"
    choice = _json(Path(cfg["out_dir"]) / f"leftover_choice{sfx(cfg)}.json")
    res, _ = leftover_report(cfg, "test", run_dir, choice)
    marker.write_text("scored once\n")
    return res


def print_leftover(res):
    if "seeds" in res and "summary" in res:
        print(f"{'system|truth':34s} contF1")
        for k, v in sorted(res["summary"].items()):
            if k.endswith("|raw"):
                print(f"{k:34s} {v['container_f1']['mean']:.3f}")
        for k, v in res["seeds"].items():
            print(f"seeds {k:22s} {v['mean']:.3f} +/- {v['std']:.3f}")
        for k, v in res["paired"].items():
            if k.endswith("|raw"):
                print(f"{k:46s} {v['diff']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] p={v['p_a_not_better']:.3f}")
    else:
        print(json.dumps(res, default=float)[:3000])


# ---------------------------------------------------------------- console
def print_summary(res):
    """Compact console summary (Colab output is cut at about 6,000 characters)."""
    if "trials" in res:
        print(f"{res['model']}: best {res['best']} AP {res['best_dev_ap']:.4f}  backbone {res.get('backbone')}"
              + (f"  trial-0 AP by backbone {res['backbone_trial0_ap']}" if "backbone_trial0_ap" in res else ""))
        for k, v in res["trials"].items():
            print(f"  {k:14s} AP {v['dev_ap']:.4f}  ({v['minutes']} min)")
        return
    if "systems" not in res:
        print(res)
        return
    print(f"{'system|truth':30s} {'contF1':>7s} {'CM':>6s} {'prec':>6s} {'edgeF1':>6s} {'pred/true':>10s}")
    for k, v in sorted(res["systems"].items()):
        if k.endswith("|raw"):
            print(f"{k:30s} {v['container_f1']['mean']:7.3f} {v['container_match']['mean']:6.3f} "
                  f"{v['container_precision']['mean']:6.3f} {v['edge_f1']['mean']:6.3f} "
                  f"{v['mean_pred_containers']:4.1f}/{v['mean_true_containers']:4.1f}")
    for k, v in res["paired"].items():
        if k.endswith("|raw"):
            print(f"{k:42s} {v['diff']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] p={v['p_a_not_better']:.3f}")
    for k, v in res["learned"].items():
        ap = next(x for kk, x in v.items() if kk.endswith("_cand_ap"))
        lt = f" leaf type macro {v['leaf_type']['macro_f1_types']:.3f}" if "leaf_type" in v else ""
        print(f"{k:12s} AP {ap:.3f} thr {v['threshold']:.2f} {v['trial']}{lt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["build", "embed", "tune", "gbdt", "dev", "test", "seeds",
                                                               "gbdt_refit", "leftover_dev", "leftover_test"])
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--model", help="tune / gbdt: which model")
    ap.add_argument("--smoke", action="store_true", help="laptop check on the first N screens (not a result)")
    ap.add_argument("--confirm-test", action="store_true", help="required for the test stage (run once)")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.smoke)
    if a.stage in ("test", "leftover_test"):          # checked before a run folder is made
        assert a.confirm_test, "the test stage runs once: pass --confirm-test"
        mk = "TEST_DONE" if a.stage == "test" else "LEFTOVER_TEST_DONE"
        assert not (Path(cfg["out_dir"]) / f"{mk}{sfx(cfg)}").exists(), \
            "the test split was already scored once; it is not run again"
    tag ={"build": f"_{a.split}", "embed": f"_{a.split}", "tune": f"_{a.model}", "gbdt": f"_{a.model}",
           "seeds": f"_{a.model}"}.get(a.stage, "")
    run_dir = start_run(f"structure_{a.stage}{tag}{sfx(cfg)}", {**cfg, "stage": a.stage, "split": a.split, "model": a.model},
                        root=cfg["runs_root"])
    res = {"build": lambda: stage_build(cfg, a.split), "embed": lambda: stage_embed(cfg, a.split),
           "tune": lambda: stage_tune(cfg, a.model), "gbdt": lambda: stage_gbdt(cfg, a.model),
           "dev": lambda: stage_dev(cfg, run_dir), "test": lambda: stage_test(cfg, run_dir),
           "seeds": lambda: stage_seeds(cfg, a.model), "gbdt_refit": lambda: stage_gbdt_refit(cfg),
           "leftover_dev": lambda: stage_leftover_dev(cfg, run_dir),
           "leftover_test": lambda: stage_leftover_test(cfg, run_dir)}[a.stage]()
    finish_run(run_dir, res)
    if a.stage.startswith("leftover") or a.stage in ("seeds", "gbdt_refit"):
        print_leftover(res)
    else:
        print_summary(res)
    print("run dir:", run_dir)
