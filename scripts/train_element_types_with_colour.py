"""Model 2 type head (Model 1 + colour features from the screenshot crop), its comparators and tests.
Three stages, each its own run folder (same pattern as scripts/train_element_types_metadata_mlp.py):

  tune  20-trial random search on clean dev, Model 2 inputs; trial 0 = Model 1's tuned settings
  dev   model2 (best tuned settings) seeds 0-2; model2 with Model 1 settings (pure colour effect);
        box + colour with Model 1 settings (colour-only row, pairs with Model 1's box row); trees on Model 2
        inputs with Model 1's trees settings; Model 1 checkpoints rescored on the same rows; paired tests on dev
  test  the dev-stage checkpoints (and the Model 1 ones) scored ONCE on clean test; nothing is refitted

Invalid AUROC (threshold-free, secondary) = AUROC of p(invalid) for invalid vs all other elements, CI by
resampling screens. Run: python scripts/train_element_types_with_colour.py --config configs/train_element_types_with_colour.yaml --stage tune|dev|test
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
from sklearn.metrics import roc_auc_score

from uistruct.data.features import FEATURE_SETS, Encoder, load_elements
from uistruct.eval.metrics import TYPES, type_metrics
from uistruct.eval.stats import bootstrap_rows, bootstrap_stat, paired_bootstrap_stat
from uistruct.models import baselines as B
from uistruct.models.type_mlp import TypeMLP
from uistruct.runlog import finish_run, start_run
from uistruct.train.type_head import invalid_f1, macro_types, per_screen_confusion, predict, sample_hp, train_mlp

LABELS = {int(k): v.strip() for k, v in (l.split(":") for l in open("data/raw/clay/label_map.txt"))}
_DF = None


def data(cfg):
    global _DF
    if _DF is None:
        _DF = load_elements(cfg["data_dir"], colour=True)
    return _DF


def model1_settings(cfg):
    """Model 1's tuned MLP settings and best trees settings, read from the Model 1 tune run."""
    tm = json.load(open(Path(cfg["element_types_mlp_tune_run"]) / "metrics.json"))
    m1_cfg = yaml.safe_load(open(Path(cfg["element_types_mlp_tune_run"]) / "config.yaml"))
    mlp = {k: tm["best_mlp"][k] for k in list(m1_cfg["mlp_space"]) + list(m1_cfg["mlp_fixed"])}
    gb = {k: tm["best_gbdt"][k] for k in m1_cfg["gbdt_space"]}
    cw = gb["class_weight"]
    gb["class_weight"] = None if cw is None or cw == "none" or (isinstance(cw, float) and np.isnan(cw)) else cw
    return mlp, gb


# ---------------------------------------------------------------- one MLP job (worker process)
def mlp_job(job):
    torch.set_num_threads(job["threads"])
    cfg, hp = job["cfg"], job["hp"]
    df = data(cfg)
    num, use_name = FEATURE_SETS[job["features"]]
    hp = {**hp, "use_name": use_name and hp.get("use_name", True), "use_pieces": use_name and hp["use_pieces"]}
    tr, dv = df[df.split_clean == "train"], df[df.split_clean == "dev"]
    enc = Encoder(num, cfg["min_count"]).fit(tr)
    model, hist = train_mlp(hp, enc, enc.transform(tr), enc.transform(dv), dv.screen_id.to_numpy(), job["seed"],
                            log=lambda s: None)
    prob = predict(model, enc.transform(dv))
    if job.get("save"):
        Path(job["save"]).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "hp": hp, "features": job["features"]}, job["save"] + ".pt")
        pickle.dump(enc, open(job["save"] + "_enc.pkl", "wb"))
    return {"name": job["name"], "hp": hp, "hist": hist, "dev_prob": prob,
            "best_epoch": int(np.argmax([h["macro_f1_types"] for h in hist]) + 1)}


def run_pool(jobs, workers):
    with ProcessPoolExecutor(workers) as ex:
        out = []
        for r in ex.map(mlp_job, jobs):
            best = max(h["macro_f1_types"] for h in r["hist"])
            print(f"{r['name']:28s} dev macro {best:.4f} (epoch {r['best_epoch']})", flush=True)
            out.append(r)
    return out


def load_mlp(path, df):
    st = torch.load(path + ".pt")
    enc = pickle.load(open(path + "_enc.pkl", "rb"))
    hp = st["hp"]
    m = TypeMLP(len(enc.numeric), len(enc.names) + 2, len(enc.pieces) + 2, use_name=hp["use_name"],
                use_pieces=hp["use_pieces"], emb_dim=hp["emb_dim"], hidden=hp["hidden"], layers=hp["layers"],
                dropout=hp["dropout"])
    m.load_state_dict(st["state"])
    return predict(m, enc.transform(df))


# ---------------------------------------------------------------- scoring
PAIRS = [("model2_s0", "model1_s0"), ("model2_m1set_s0", "model1_s0"), ("model2_s0", "model2_m1set_s0"),
         ("model2_s0", "gbdt2"), ("gbdt2", "gbdt"), ("box_colour_s0", "box_s0"), ("model2_s0", "lookup")]


def evaluate(sid, y, probs, cfg):
    """probs: {name: (n, 24) probabilities, or (n,) hard labels (lookup)}."""
    pred = {k: (p.argmax(1) if p.ndim == 2 else p) for k, p in probs.items()}
    inv = (y == 0).astype(int)
    nb, sd = cfg["n_boot"], cfg["seed"]
    out, cms = {"scores": {}}, {}
    for k, p in pred.items():
        cms[k] = per_screen_confusion(sid, y, p)
        out["scores"][k] = {"macro_f1_types": bootstrap_stat(cms[k], macro_types, nb, sd),
                            "invalid_f1": bootstrap_stat(cms[k], invalid_f1, nb, sd)}
        if probs[k].ndim == 2:
            pk = probs[k][:, 0]
            out["scores"][k]["invalid_auroc"] = bootstrap_rows(sid, lambda i, pk=pk: roc_auc_score(inv[i], pk[i]), nb, sd)
    pairs = [(a, b) for a, b in PAIRS if a in cms and b in cms]
    out["paired_macro"] = {f"{a}_vs_{b}": paired_bootstrap_stat(cms[a], cms[b], macro_types, nb, sd) for a, b in pairs}
    out["paired_invalid_f1"] = {f"{a}_vs_{b}": paired_bootstrap_stat(cms[a], cms[b], invalid_f1, nb, sd) for a, b in pairs}
    out["paired_invalid_auroc"] = {}
    for a, b in pairs:
        if probs[a].ndim == 2 and probs[b].ndim == 2:
            pa, pb = probs[a][:, 0], probs[b][:, 0]
            out["paired_invalid_auroc"][f"{a}_vs_{b}"] = bootstrap_rows(
                sid, lambda i: roc_auc_score(inv[i], pa[i]), nb, sd, other=lambda i: roc_auc_score(inv[i], pb[i]))
    names = ["INVALID(BACKGROUND)"] + [LABELS[c] for c in TYPES]

    def pc(p):
        m = type_metrics(y, p)
        return dict(zip(names, [round(m["invalid_f1"], 4)] + [m["per_class_f1"][c] for c in TYPES]))
    a, b = pc(pred["model2_s0"]), pc(pred["model1_s0"])
    out["per_class_f1"] = {n: {"model2": a[n], "model1": b[n],
                               "delta": None if a[n] is None or b[n] is None else round(a[n] - b[n], 4)} for n in names}
    return out


# ---------------------------------------------------------------- stages
def stage_tune(cfg, run_dir):
    m1, _ = model1_settings(cfg)
    rng = np.random.default_rng(cfg["seed"])
    space = cfg["mlp_space"]
    hps = [{k: m1[k] for k in space}] + [sample_hp(space, rng) for _ in range(cfg["n_mlp_trials"] - 1)]
    jobs = [{"name": f"mlp_trial_{i:02d}", "cfg": cfg, "hp": {**cfg["mlp_fixed"], **hp}, "features": "model2",
             "seed": cfg["seed"], "threads": cfg["threads_per_worker"]} for i, hp in enumerate(hps)]
    res = run_pool(jobs, cfg["workers"])
    rows = [{"trial": r["name"], **r["hp"], "best_epoch": r["best_epoch"],
             **max(r["hist"], key=lambda h: h["macro_f1_types"])} for r in res]
    mlp = pd.DataFrame(rows).sort_values("macro_f1_types", ascending=False)
    mlp.to_csv(run_dir / "trials_mlp.csv", index=False)
    json.dump({r["name"]: r["hist"] for r in res}, open(run_dir / "histories_mlp.json", "w"), indent=1)
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in mlp.iloc[0].items()}
    return {"best_mlp": best, "n_mlp_trials": len(mlp),
            "trial0_model1_settings_dev_macro": float(mlp[mlp.trial == "mlp_trial_00"].macro_f1_types.iloc[0])}


def stage_dev(cfg, run_dir):
    df = data(cfg)
    m1, gb_hp = model1_settings(cfg)
    tm = json.load(open(Path(cfg["tune_run"]) / "metrics.json"))
    m2 = {k: tm["best_mlp"][k] for k in list(cfg["mlp_space"]) + list(cfg["mlp_fixed"])}
    ck = run_dir / "checkpoints"
    jobs = [{"name": "model2_s0", "features": "model2", "seed": 0, "hp": m2},
            {"name": "model2_m1set_s0", "features": "model2", "seed": 0, "hp": m1},
            {"name": "box_colour_s0", "features": "box_colour", "seed": 0, "hp": m1}]
    jobs += [{"name": f"model2_s{s}", "features": "model2", "seed": s, "hp": m2} for s in cfg["extra_seeds"]]
    for j in jobs:
        j.update(cfg=cfg, threads=cfg["threads_per_worker"], save=str(ck / j["name"]))
    res = {r["name"]: r for r in run_pool(jobs, cfg["workers"])}

    tr, dv = df[df.split_clean == "train"], df[df.split_clean == "dev"]
    sid, y = dv.screen_id.to_numpy(), dv.label.to_numpy()
    probs = {k: r["dev_prob"] for k, r in res.items()}
    gi = B.GBDTInputs(FEATURE_SETS["model2"][0]).fit(tr)
    g = B.gbdt({**gb_hp, "max_iter": cfg["gbdt_dev_max_iter"]}, cfg["seed"]).set_params(
        categorical_features=gi.categorical_mask).fit(gi.transform(tr), tr.label.to_numpy())
    print("gbdt2 done", g.n_iter_, flush=True)
    probs["gbdt2"] = g.predict_proba(gi.transform(dv))
    pickle.dump({"model": g, "inputs": gi}, open(ck / "gbdt2.pkl", "wb"))
    m1_ck = Path(cfg["element_types_mlp_dev_run"]) / "checkpoints"
    for n in ["model1_s0", "model1_s1", "model1_s2", "box_s0"]:
        probs[n] = load_mlp(str(m1_ck / n), dv)
    g1 = pickle.load(open(m1_ck / "gbdt.pkl", "rb"))
    probs["gbdt"] = g1["model"].predict_proba(g1["inputs"].transform(dv))
    probs["lookup"] = B.lookup(tr, dv)
    np.savez_compressed(run_dir / "dev_probs.npz", screen_id=sid, y=y, **probs)

    out = {"settings": {"model2": m2, "model1": m1, "gbdt": gb_hp},
           "best_epoch": {k: r["best_epoch"] for k, r in res.items()}, "gbdt2_n_iter": int(g.n_iter_),
           **evaluate(sid, y, probs, cfg)}
    for m in ["model1", "model2"]:
        s = [out["scores"][f"{m}_s{i}"]["macro_f1_types"]["value"] for i in [0, 1, 2]]
        out[f"{m}_seeds_macro"] = {"mean": float(np.mean(s)), "std": float(np.std(s, ddof=1)), "values": s}
    json.dump({k: r["hist"] for k, r in res.items()}, open(run_dir / "histories.json", "w"), indent=1)
    return out


def stage_test(cfg, run_dir):
    df = data(cfg)
    tr, te = df[df.split_clean == "train"], df[df.split_clean == "test"]
    sid, y = te.screen_id.to_numpy(), te.label.to_numpy()
    ck, m1_ck = Path(cfg["dev_run"]) / "checkpoints", Path(cfg["element_types_mlp_dev_run"]) / "checkpoints"
    probs = {}
    for n in ["model2_s0", "model2_m1set_s0", "box_colour_s0"] + [f"model2_s{s}" for s in cfg["extra_seeds"]]:
        probs[n] = load_mlp(str(ck / n), te)
    for n in ["model1_s0", "model1_s1", "model1_s2", "box_s0"]:
        probs[n] = load_mlp(str(m1_ck / n), te)
    for n, d in [("gbdt2", ck / "gbdt2.pkl"), ("gbdt", m1_ck / "gbdt.pkl")]:
        g = pickle.load(open(d, "rb"))
        probs[n] = g["model"].predict_proba(g["inputs"].transform(te))
    probs["lookup"] = B.lookup(tr, te)
    np.savez_compressed(run_dir / "test_probs.npz", screen_id=sid, y=y, **probs)
    out = evaluate(sid, y, probs, cfg)
    for m in ["model1", "model2"]:
        names = [f"{m}_s{i}" for i in [0, 1, 2]]
        out[f"{m}_seeds"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for k, v in {
            "macro": [out["scores"][n]["macro_f1_types"]["value"] for n in names],
            "invalid_f1": [out["scores"][n]["invalid_f1"]["value"] for n in names],
            "invalid_auroc": [out["scores"][n]["invalid_auroc"]["value"] for n in names]}.items()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["tune", "dev", "test"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.set_num_threads(cfg["threads_main"])
    run_dir = start_run(f"element_types_colour_{args.stage}", {**cfg, "stage": args.stage})
    res = {"tune": stage_tune, "dev": stage_dev, "test": stage_test}[args.stage](cfg, run_dir)
    finish_run(run_dir, res)
    print(json.dumps(res, indent=1, default=float)[:8000])
    print("run dir:", run_dir)
