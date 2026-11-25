"""Model 1 type head (geometry + raw tree + flags + class name), its baselines, tuning, ablation rows,
learning curve and seeds. Three stages, each its own run folder:

  tune  random search on clean dev: MLP (trial 0 = proposal-era settings) and gradient-boosted trees
  dev   best settings: ablation rows box / box_tree_flags / model1 (seed 0), learning curve 25/50/75%
        (nested company subsets, seed 0), seeds 1-2 at 100%, trees model, majority and lookup; paired
        bootstrap tests on dev; checkpoints saved (git-ignored)
  test  the dev-stage checkpoints scored ONCE on clean test; nothing is refitted

Run: python scripts/train_element_types_metadata_mlp.py --config configs/train_element_types_metadata_mlp.yaml --stage tune|dev|test
"""
import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from uistruct.data.features import FEATURE_SETS, Encoder, load_elements
from uistruct.data.splits import nested_company_subsets
from uistruct.eval.metrics import TYPES
from uistruct.eval.stats import bootstrap_stat, paired_bootstrap_stat
from uistruct.models import baselines as B
from uistruct.runlog import finish_run, start_run
from uistruct.train.type_head import (invalid_f1, macro_types, per_screen_confusion, predict, sample_hp,
                                      scores, train_mlp)

LABELS = {int(k): v.strip() for k, v in (l.split(":") for l in open("data/raw/clay/label_map.txt"))}
_DF = None


def data(cfg):
    global _DF
    if _DF is None:
        _DF = load_elements(cfg["data_dir"])
    return _DF


def train_rows(df, frac_ids):
    tr = df[df.split_clean == "train"]
    return tr if frac_ids is None else tr[tr.screen_id.isin(frac_ids)]


# ---------------------------------------------------------------- one MLP job (runs in a worker process)
def mlp_job(job):
    torch.set_num_threads(job["threads"])
    cfg, hp = job["cfg"], job["hp"]
    df = data(cfg)
    num, use_name = FEATURE_SETS[job["features"]]
    hp = {**hp, "use_name": use_name and hp.get("use_name", True), "use_pieces": use_name and hp["use_pieces"]}
    tr, dv = train_rows(df, job.get("train_ids")), df[df.split_clean == "dev"]
    enc = Encoder(num, cfg["min_count"]).fit(tr)
    logs = []
    model, hist = train_mlp(hp, enc, enc.transform(tr), enc.transform(dv), dv.screen_id.to_numpy(), job["seed"],
                            log=logs.append)
    p = predict(model, enc.transform(dv)).argmax(1)
    if job.get("save"):
        Path(job["save"]).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "hp": hp, "features": job["features"]}, job["save"] + ".pt")
        pickle.dump(enc, open(job["save"] + "_enc.pkl", "wb"))
    return {"name": job["name"], "hp": hp, "hist": hist, "dev_pred": p, "n_train": len(tr),
            "best_epoch": int(np.argmax([h["macro_f1_types"] for h in hist]) + 1)}


def run_pool(jobs, workers):
    with ProcessPoolExecutor(workers) as ex:
        out = []
        for r in ex.map(mlp_job, jobs):
            best = max(h["macro_f1_types"] for h in r["hist"])
            print(f"{r['name']:28s} dev macro {best:.4f} (epoch {r['best_epoch']}, n_train {r['n_train']})", flush=True)
            out.append(r)
    return out


def ci(sids, y, p, n_boot, seed):
    cm = per_screen_confusion(sids, y, p)
    return cm, {"macro_f1_types": bootstrap_stat(cm, macro_types, n_boot, seed),
                "invalid_f1": bootstrap_stat(cm, invalid_f1, n_boot, seed)}


def per_class(y, p):
    from uistruct.eval.metrics import type_metrics
    m = type_metrics(y, p)
    return {LABELS[c]: m["per_class_f1"][c] for c in TYPES} | {"INVALID(BACKGROUND)": round(m["invalid_f1"], 4),
                                                               "accuracy_valid": round(m["accuracy_valid"], 4)}


# ---------------------------------------------------------------- stages
def stage_tune(cfg, run_dir):
    rng = np.random.default_rng(cfg["seed"])
    hps = [dict(cfg["mlp_anchor"])] + [sample_hp(cfg["mlp_space"], rng) for _ in range(cfg["n_mlp_trials"] - 1)]
    jobs = [{"name": f"mlp_trial_{i:02d}", "cfg": cfg, "hp": {**cfg["mlp_fixed"], **hp}, "features": "model1",
             "seed": cfg["seed"], "threads": cfg["threads_per_worker"]} for i, hp in enumerate(hps)]
    res = run_pool(jobs, cfg["workers"])
    rows = [{"trial": r["name"], **{k: v for k, v in r["hp"].items()}, "best_epoch": r["best_epoch"],
             **max(r["hist"], key=lambda h: h["macro_f1_types"])} for r in res]
    mlp = pd.DataFrame(rows).sort_values("macro_f1_types", ascending=False)
    mlp.to_csv(run_dir / "trials_mlp.csv", index=False)
    json.dump({r["name"]: r["hist"] for r in res}, open(run_dir / "histories_mlp.json", "w"), indent=1)

    df = data(cfg)
    num, _ = FEATURE_SETS["model1"]
    tr, dv = df[df.split_clean == "train"], df[df.split_clean == "dev"]
    gi = B.GBDTInputs(num).fit(tr)
    X, Xd = gi.transform(tr), gi.transform(dv)
    grows = []
    for i in range(cfg["n_gbdt_trials"]):
        hp = sample_hp(cfg["gbdt_space"], rng)
        hp["class_weight"] = None if hp["class_weight"] == "none" else hp["class_weight"]
        g = B.gbdt(hp, cfg["seed"]).set_params(categorical_features=gi.categorical_mask)
        g.fit(X, tr.label.to_numpy())
        sc = scores(per_screen_confusion(dv.screen_id.to_numpy(), dv.label.to_numpy(), g.predict(Xd)).sum(0))
        grows.append({"trial": f"gbdt_trial_{i:02d}", **hp, "n_iter": int(g.n_iter_), **sc})
        print(f"gbdt_trial_{i:02d} {hp} -> {sc} ({g.n_iter_} it)", flush=True)
    gb = pd.DataFrame(grows).sort_values("macro_f1_types", ascending=False)
    gb.to_csv(run_dir / "trials_gbdt.csv", index=False)
    best_mlp = {k: (v.item() if hasattr(v, "item") else v) for k, v in mlp.iloc[0].items()}
    best_gb = {k: (v.item() if hasattr(v, "item") else v) for k, v in gb.iloc[0].items()}
    return {"best_mlp": best_mlp, "best_gbdt": best_gb, "anchor_mlp_dev_macro": float(mlp[mlp.trial == "mlp_trial_00"].macro_f1_types.iloc[0]),
            "n_mlp_trials": len(mlp), "n_gbdt_trials": len(gb)}


def best_settings(cfg):
    tm = json.load(open(Path(cfg["tune_run"]) / "metrics.json"))
    keys = list(cfg["mlp_space"]) + list(cfg["mlp_fixed"])
    mlp = {k: tm["best_mlp"][k] for k in keys}
    gk = list(cfg["gbdt_space"])
    gb = {k: tm["best_gbdt"][k] for k in gk}
    return mlp, gb


def stage_dev(cfg, run_dir):
    df = data(cfg)
    mlp_hp, gb_hp = best_settings(cfg)
    ck = run_dir / "checkpoints"
    screens = pd.read_parquet(f"{cfg['data_dir']}/splits.parquet")
    subsets = nested_company_subsets(screens[screens.split_clean == "train"], cfg["lc_fracs"], cfg["seed"])
    jobs = []
    for fs in ["box", "box_tree_flags", "model1"]:
        jobs.append({"name": f"{fs}_s0", "features": fs, "seed": 0})
    for f in cfg["lc_fracs"]:
        jobs.append({"name": f"model1_frac{int(f * 100)}_s0", "features": "model1", "seed": 0, "train_ids": subsets[f]})
    for s in cfg["extra_seeds"]:
        jobs.append({"name": f"model1_s{s}", "features": "model1", "seed": s})
    for j in jobs:
        j.update(cfg=cfg, hp=mlp_hp, threads=cfg["threads_per_worker"], save=str(ck / j["name"]))
    res = {r["name"]: r for r in run_pool(jobs, cfg["workers"])}

    tr, dv = df[df.split_clean == "train"], df[df.split_clean == "dev"]
    sid, y = dv.screen_id.to_numpy(), dv.label.to_numpy()
    preds = {k: r["dev_pred"] for k, r in res.items()}
    preds["majority"], preds["lookup"] = B.majority(tr, dv), B.lookup(tr, dv)
    num, _ = FEATURE_SETS["model1"]
    gi = B.GBDTInputs(num).fit(tr)
    cw = gb_hp["class_weight"]
    cw = None if cw is None or cw == "none" or (isinstance(cw, float) and np.isnan(cw)) else cw
    hp = {**gb_hp, "class_weight": cw, "max_iter": cfg["gbdt_dev_max_iter"]}
    g = B.gbdt(hp, cfg["seed"]).set_params(categorical_features=gi.categorical_mask).fit(gi.transform(tr), tr.label.to_numpy())
    preds["gbdt"] = g.predict(gi.transform(dv))
    pickle.dump({"model": g, "inputs": gi}, open(ck / "gbdt.pkl", "wb"))
    np.savez_compressed(run_dir / "dev_predictions.npz", screen_id=sid, y=y, **preds)

    out = {"settings": {"mlp": mlp_hp, "gbdt": gb_hp}, "lc_train_screens": {str(f): len(v) for f, v in subsets.items()},
           "lc_train_elements": {k: r["n_train"] for k, r in res.items()}, "dev": {}, "best_epoch": {k: r["best_epoch"] for k, r in res.items()}}
    cms = {}
    for k, p in preds.items():
        cms[k], out["dev"][k] = ci(sid, y, p, cfg["n_boot"], cfg["seed"])
    seeds = [out["dev"][k]["macro_f1_types"]["value"] for k in ["model1_s0"] + [f"model1_s{s}" for s in cfg["extra_seeds"]]]
    out["dev_model1_seeds"] = {"mean": float(np.mean(seeds)), "std": float(np.std(seeds, ddof=1)), "values": seeds}
    out["dev_paired"] = {f"{a}_vs_{b}": paired_bootstrap_stat(cms[a], cms[b], macro_types, cfg["n_boot"], cfg["seed"])
                         for a, b in cfg["paired"]}
    out["dev_learning_curve"] = {str(f): out["dev"][f"model1_frac{int(f * 100)}_s0"]["macro_f1_types"]["value"] for f in cfg["lc_fracs"]}
    out["dev_learning_curve"]["1.0"] = out["dev"]["model1_s0"]["macro_f1_types"]["value"]
    json.dump({k: r["hist"] for k, r in res.items()}, open(run_dir / "histories.json", "w"), indent=1)
    return out


def stage_test(cfg, run_dir):
    df = data(cfg)
    tr, te = df[df.split_clean == "train"], df[df.split_clean == "test"]
    sid, y = te.screen_id.to_numpy(), te.label.to_numpy()
    ck = Path(cfg["dev_run"]) / "checkpoints"
    names = ["box_s0", "box_tree_flags_s0", "model1_s0"] + [f"model1_s{s}" for s in cfg["extra_seeds"]]
    from uistruct.models.type_mlp import TypeMLP
    preds = {}
    for n in names:
        st = torch.load(ck / f"{n}.pt")
        enc = pickle.load(open(ck / f"{n}_enc.pkl", "rb"))
        hp = st["hp"]
        m = TypeMLP(len(enc.numeric), len(enc.names) + 2, len(enc.pieces) + 2, use_name=hp["use_name"],
                    use_pieces=hp["use_pieces"], emb_dim=hp["emb_dim"], hidden=hp["hidden"], layers=hp["layers"],
                    dropout=hp["dropout"])
        m.load_state_dict(st["state"])
        preds[n] = predict(m, enc.transform(te)).argmax(1)
    gb = pickle.load(open(ck / "gbdt.pkl", "rb"))
    preds["gbdt"] = gb["model"].predict(gb["inputs"].transform(te))
    preds["majority"], preds["lookup"] = B.majority(tr, te), B.lookup(tr, te)
    np.savez_compressed(run_dir / "test_predictions.npz", screen_id=sid, y=y, **preds)
    out, cms = {"test": {}}, {}
    for k, p in preds.items():
        cms[k], out["test"][k] = ci(sid, y, p, cfg["n_boot"], cfg["seed"])
    seeds = [out["test"][n]["macro_f1_types"]["value"] for n in names if n.startswith("model1")]
    inv = [out["test"][n]["invalid_f1"]["value"] for n in names if n.startswith("model1")]
    out["test_model1_seeds"] = {"macro_mean": float(np.mean(seeds)), "macro_std": float(np.std(seeds, ddof=1)),
                                "invalid_mean": float(np.mean(inv)), "invalid_std": float(np.std(inv, ddof=1))}
    out["test_paired"] = {f"{a}_vs_{b}": paired_bootstrap_stat(cms[a], cms[b], macro_types, cfg["n_boot"], cfg["seed"])
                          for a, b in cfg["paired"] if a in cms and b in cms}
    out["test_per_class_f1"] = {k: per_class(y, preds[k]) for k in ["model1_s0", "gbdt", "lookup", "box_s0"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["tune", "dev", "test"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.set_num_threads(cfg["threads_main"])
    run_dir = start_run(f"element_types_mlp_{args.stage}", {**cfg, "stage": args.stage})
    res = {"tune": stage_tune, "dev": stage_dev, "test": stage_test}[args.stage](cfg, run_dir)
    finish_run(run_dir, res)
    print(json.dumps(res, indent=1, default=float)[:6000])
    print("run dir:", run_dir)
