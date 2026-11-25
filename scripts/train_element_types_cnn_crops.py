"""Model 3 type head = Model 1 + a pretrained CNN embedding of each element's crop (ResNet-18, ImageNet
weights only), and its comparators. Runs on Colab (notebooks/train_element_type_classifier_with_resnet18_crops.ipynb). Stages, each its own run folder
(saved on Google Drive, so they survive the Colab session):

  features  frozen ResNet-18 features at every pilot crop size, all clean elements, one decoding pass
  frozen    MLP with Model 1's settings on [inputs + frozen features]: model1_cnn at every size -> crop size
            (rule fixed in the config); then model2_cnn and box_cnn at that size. Dev only
  cache     cut the crop cache of each split at the chosen size (local disk; needed by tune/dev/test)
  tune      fine-tune model1_cnn (backbone + MLP) for each backbone learning rate, seed 0, dev selection
  dev       best rate: model1_cnn seeds 1-2, model2_cnn and box_cnn seed 0; dev report with Model 2's models
  test      every Model 3 checkpoint scored ONCE on clean test, with Model 2's test probabilities for paired tests

Every trained model is saved to the checkpoint folder on Drive as soon as it finishes; a stage re-run after a
disconnect skips models already saved (resumable). Names: model3 = model1_cnn fine-tuned (Model 3);
*_frozen = frozen backbone at the chosen size.

  python scripts/train_element_types_cnn_crops.py --config configs/train_element_types_cnn_crops.yaml --stage features|frozen|cache|tune|dev|test
"""
import argparse
import functools
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from uistruct.data.crops import build_cache, iter_screen_crops
from uistruct.data.features import FEATURE_SETS, Encoder, load_elements
from uistruct.eval.type_report import report
from uistruct.models.cnn import FEAT_DIM, embed, resnet18_trunk
from uistruct.runlog import finish_run, start_run
from uistruct.train import cnn_head as C

SPLITS = ("train", "dev", "test")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DF = None


def data(cfg):
    global _DF
    if _DF is None:
        _DF = load_elements(cfg["data_dir"], colour=True)
        if cfg.get("limit_screens"):     # smoke tests only: the first N screens of each split
            keep = [_DF.screen_id[_DF.split_clean == sp].unique()[:cfg["limit_screens"]] for sp in SPLITS]
            _DF = _DF[_DF.screen_id.isin(np.concatenate(keep))].reset_index(drop=True)
    return _DF


def masks(df):
    return {s: (df.split_clean == s).to_numpy() for s in SPLITS}


def model1_settings(cfg):
    """Model 1's tuned MLP settings, read from the Model 1 tune run (as for Model 2)."""
    tm = json.load(open(Path(cfg["element_types_mlp_tune_run"]) / "metrics.json"))
    m1_cfg = yaml.safe_load(open(Path(cfg["element_types_mlp_tune_run"]) / "config.yaml"))
    return {k: tm["best_mlp"][k] for k in list(m1_cfg["mlp_space"]) + list(m1_cfg["mlp_fixed"])}


@functools.lru_cache(maxsize=None)
def _latest(runs_root, stage):
    done = [p for p in Path(runs_root).glob(f"element_types_cnn_{stage}_run*") if (p / "metrics.json").exists()]
    if not done:
        raise FileNotFoundError(f"no finished Model 3 {stage} run in {runs_root}; run that stage first")
    p = max(done, key=lambda p: int(p.name.rsplit("_run", 1)[1]))
    print(f"using {p.name}", flush=True)
    return json.load(open(p / "metrics.json"))


def latest_run(cfg, stage):
    """Metrics of the newest finished run of a stage (a run cut off by a disconnect has no metrics.json)."""
    return _latest(cfg["runs_root"], stage)


def crop_size(cfg):
    if cfg.get("crop_size"):
        return int(cfg["crop_size"])
    return int(latest_run(cfg, "frozen")["chosen_size"])


def feat_path(cfg, size):
    return Path(cfg["work"]) / f"feats_{size}.npy"


def cache_path(cfg, size, split):
    return Path(cfg["work"]) / f"crops_{size}_{split}.npy"


def ckpt(cfg, name):
    return Path(cfg["ckpt_dir"]) / name


# ---------------------------------------------------------------- one model: train (or load if saved), dev probs
def fit_one(cfg, name, features, hp, seed, fine_tune, src_tr, src_dv, log=print):
    p = ckpt(cfg, name)
    if (p.parent / f"{name}_dev_prob.npy").exists():
        log(f"{name}: already saved, skipped")
        return json.load(open(p.parent / f"{name}_hist.json"))
    df, m = data(cfg), masks(data(cfg))
    num, use_name = FEATURE_SETS[features]
    hp = {**hp, "use_name": use_name and hp.get("use_name", True), "use_pieces": use_name and hp["use_pieces"]}
    tr_df, dv_df = df[m["train"]], df[m["dev"]]
    enc = Encoder(num, cfg["min_count"]).fit(tr_df)
    tr, dv = enc.transform(tr_df), enc.transform(dv_df)
    log(f"{name}: training ({features}, {'fine-tuned' if fine_tune else 'frozen'}, seed {seed})")
    t0 = time.time()
    model, hist = C.train(hp, enc, tr, dv, src_tr, src_dv, dv_df.screen_id.to_numpy(), seed, DEV, fine_tune, log,
                          cfg.get("pretrained", True))
    prob = C.predict(model, dv, src_dv, DEV)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "hp": hp, "features": features, "fine_tune": fine_tune,
                "seed": seed}, str(p) + ".pt")
    pickle.dump(enc, open(str(p) + "_enc.pkl", "wb"))
    out = {"name": name, "features": features, "fine_tune": fine_tune, "seed": seed, "hp": hp, "hist": hist,
           "minutes": round((time.time() - t0) / 60, 1),
           "best_epoch": int(np.argmax([h["macro_f1_types"] for h in hist]) + 1)}
    json.dump(out, open(str(p) + "_hist.json", "w"), indent=1, default=float)
    np.save(str(p) + "_dev_prob.npy", prob.astype(np.float16))      # written last: marks the model as done
    return out


def load_one(cfg, name, split_df, src):
    st = torch.load(str(ckpt(cfg, name)) + ".pt", map_location="cpu")
    enc = pickle.load(open(str(ckpt(cfg, name)) + "_enc.pkl", "rb"))
    model = C.build_model(st["hp"], enc, len(enc.numeric), st["fine_tune"], pretrained=False)  # weights loaded next
    model.load_state_dict(st["state"])
    return C.predict(model.to(DEV), enc.transform(split_df), src, DEV)


def best_dev(h):
    return max(e["macro_f1_types"] for e in h["hist"])


# ---------------------------------------------------------------- stages
def stage_features(cfg, run_dir):
    df = data(cfg)
    sizes = cfg["pilot_sizes"]
    bb = resnet18_trunk(cfg.get("pretrained", True))
    out = {s: np.lib.format.open_memmap(feat_path(cfg, s), mode="w+", dtype=np.float16, shape=(len(df), FEAT_DIM))
           for s in sizes}
    buf, start, t0 = {s: [] for s in sizes}, 0, time.time()

    def flush(end):
        for s in sizes:
            out[s][start:end] = embed(bb, np.concatenate(buf[s]), DEV, cfg["embed_batch"])
            buf[s] = []
    pending = 0
    for k, (a, b, crops) in enumerate(iter_screen_crops(df, cfg["img_dir"], sizes, cfg["crop_workers"])):
        for s in sizes:
            buf[s].append(crops[s])
        pending = b
        if pending - start >= cfg["embed_batch"] * 4:
            flush(pending)
            start = pending
        if k % 5000 == 0:
            print(f"  features: row {b:,} / {len(df):,} ({time.time() - t0:.0f}s)", flush=True)
    if pending > start:
        flush(pending)
    for s in sizes:
        out[s].flush()
    sec = time.time() - t0
    return {"n_elements": len(df), "sizes": sizes, "seconds": round(sec), "elements_per_s": len(df) / sec}


def stage_frozen(cfg, run_dir):
    df, m = data(cfg), masks(data(cfg))
    hp = {**model1_settings(cfg), **cfg["frozen_fixed"]}
    res = {}
    for s in cfg["pilot_sizes"]:
        f = np.load(feat_path(cfg, s), mmap_mode="r")
        src_tr, src_dv = C.Source("feat", f[m["train"]], DEV), C.Source("feat", f[m["dev"]], DEV)
        res[f"model1_cnn_frozen_{s}"] = fit_one(cfg, f"model1_cnn_frozen_{s}", "model1_cnn", hp, cfg["seed"],
                                                False, src_tr, src_dv)
        del src_tr, src_dv
        torch.cuda.empty_cache()
    dev = {s: best_dev(res[f"model1_cnn_frozen_{s}"]) for s in cfg["pilot_sizes"]}
    top = max(dev.values())
    chosen = min(s for s, v in dev.items() if v >= top - cfg["size_tie"])
    print("dev macro by size:", dev, "-> chosen", chosen, flush=True)
    f = np.load(feat_path(cfg, chosen), mmap_mode="r")
    src_tr, src_dv = C.Source("feat", f[m["train"]], DEV), C.Source("feat", f[m["dev"]], DEV)
    for feats, name in [("model2_cnn", "model2cnn_frozen"), ("box_cnn", "boxcnn_frozen")]:
        res[name] = fit_one(cfg, f"{name}_{chosen}", feats, hp, cfg["seed"], False, src_tr, src_dv)
    return {"dev_macro_by_size": dev, "chosen_size": chosen, "settings": hp,
            "runs": {k: {"dev_macro": best_dev(v), "best_epoch": v["best_epoch"], "minutes": v.get("minutes"),
                         "hist": v["hist"]} for k, v in res.items()}}


def stage_cache(cfg, run_dir):
    df, m = data(cfg), masks(data(cfg))
    s = crop_size(cfg)
    out = {"size": s}
    for sp in SPLITS:
        t0 = time.time()
        build_cache(df[m[sp]], cfg["img_dir"], s, cache_path(cfg, s, sp), cfg["crop_workers"],
                    log=lambda x: print(x, flush=True))
        out[sp] = {"rows": int(m[sp].sum()), "seconds": round(time.time() - t0),
                   "gb": round(cache_path(cfg, s, sp).stat().st_size / 1e9, 2)}
        print(sp, out[sp], flush=True)
    return out


def crop_sources(cfg, *splits):
    s = crop_size(cfg)
    return [C.Source("crop", np.load(cache_path(cfg, s, sp), mmap_mode="r"), DEV) for sp in splits]


def stage_tune(cfg, run_dir):
    hp0 = {**model1_settings(cfg), **cfg["ft_fixed"]}
    src_tr, src_dv = crop_sources(cfg, "train", "dev")
    res = {}
    for i, lr in enumerate(cfg["lr_backbone_trials"]):
        res[f"model3_lr{i}"] = fit_one(cfg, f"model3_lr{i}", "model1_cnn", {**hp0, "lr_backbone": lr},
                                       cfg["seed"], True, src_tr, src_dv)
    best = max(res, key=lambda k: best_dev(res[k]))
    return {"crop_size": crop_size(cfg), "best": best, "best_lr_backbone": res[best]["hp"]["lr_backbone"],
            "trials": {k: {"lr_backbone": v["hp"]["lr_backbone"], "dev_macro": best_dev(v),
                           "best_epoch": v["best_epoch"], "minutes": v.get("minutes"), "hist": v["hist"]}
                       for k, v in res.items()}}


def dev_names(cfg):
    """{name: (features, fine_tune)} of every Model 3 run scored in the dev and test reports."""
    s = crop_size(cfg)
    best = latest_run(cfg, "tune")["best"]
    names = {best: ("model1_cnn", True), f"model1_cnn_frozen_{s}": ("model1_cnn", False),
             f"model2cnn_frozen_{s}": ("model2_cnn", False), f"boxcnn_frozen_{s}": ("box_cnn", False)}
    names.update({f"model3_s{x}": ("model1_cnn", True) for x in cfg["extra_seeds"]})
    names.update({"model2cnn_s0": ("model2_cnn", True), "boxcnn_s0": ("box_cnn", True)})
    return best, names


def short(cfg, name, best):
    """Report names: the best tuning trial is model3_s0; frozen models lose the size suffix."""
    s = crop_size(cfg)
    return {best: "model3_s0", f"model1_cnn_frozen_{s}": "model3_frozen", f"model2cnn_frozen_{s}": "model2cnn_frozen",
            f"boxcnn_frozen_{s}": "boxcnn_frozen"}.get(name, name)


PAIRS = [("model3_s0", "model2_s0"), ("model3_s0", "model1_s0"), ("model3_s0", "gbdt2"),
         ("model3_s0", "model3_frozen"), ("model2cnn_s0", "model3_s0"), ("model2cnn_s0", "model2_s0"),
         ("boxcnn_s0", "box_colour_s0"), ("boxcnn_s0", "box_s0"), ("model3_frozen", "model2_s0"),
         ("model2cnn_frozen", "model2_s0"), ("boxcnn_frozen", "box_colour_s0")]
EARLIER_MODELS = ["model2_s0", "model2_s1", "model2_s2", "model1_s0", "model1_s1", "model1_s2", "gbdt2", "gbdt",
              "box_colour_s0", "box_s0", "lookup"]


def labels(cfg):
    return {int(k): v.strip() for k, v in (l.split(":") for l in open(Path(cfg["extra_dir"]) / "label_map.txt"))}


def with_colour_model(cfg, split, sid, y, probs):
    z = np.load(Path(cfg["extra_dir"]) / f"element_types_colour_{split}_probs.npz")
    keep = np.isin(z["screen_id"], sid) if cfg.get("limit_screens") else slice(None)
    assert (z["screen_id"][keep] == sid).all() and (z["y"][keep] == y).all(), "Model 2 rows do not match Model 3 rows"
    return {**probs, **{k: z[k][keep] for k in EARLIER_MODELS}}


def seeds_summary(out):
    names = ["model3_s0", "model3_s1", "model3_s2"]
    if not all(n in out["scores"] for n in names):
        return None
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v} for k, v in {
        "macro": [out["scores"][n]["macro_f1_types"]["value"] for n in names],
        "invalid_f1": [out["scores"][n]["invalid_f1"]["value"] for n in names],
        "invalid_auroc": [out["scores"][n]["invalid_auroc"]["value"] for n in names]}.items()}


def stage_dev(cfg, run_dir):
    df, m = data(cfg), masks(data(cfg))
    tm = latest_run(cfg, "tune")
    hp = {**model1_settings(cfg), **cfg["ft_fixed"], "lr_backbone": tm["best_lr_backbone"]}
    src_tr, src_dv = crop_sources(cfg, "train", "dev")
    res = {}
    for x in cfg["extra_seeds"]:
        res[f"model3_s{x}"] = fit_one(cfg, f"model3_s{x}", "model1_cnn", hp, x, True, src_tr, src_dv)
    for feats, name in [("model2_cnn", "model2cnn_s0"), ("box_cnn", "boxcnn_s0")]:
        res[name] = fit_one(cfg, name, feats, hp, cfg["seed"], True, src_tr, src_dv)
    best, names = dev_names(cfg)
    dv = df[m["dev"]]
    sid, y = dv.screen_id.to_numpy(), dv.label.to_numpy()
    probs = {short(cfg, n, best): np.load(ckpt(cfg, n).parent / f"{n}_dev_prob.npy").astype(np.float32)
             for n in names}
    probs = with_colour_model(cfg, "dev", sid, y, probs)
    out = {"settings": hp, "runs": {k: {"dev_macro": best_dev(v), "best_epoch": v["best_epoch"],
                                        "minutes": v.get("minutes"), "hist": v["hist"]} for k, v in res.items()},
           **report(sid, y, probs, PAIRS, labels(cfg), ("model3_s0", "model2_s0"), cfg["n_boot"], cfg["seed"])}
    out["model3_seeds"] = seeds_summary(out)
    return out


def stage_test(cfg, run_dir):
    df, m = data(cfg), masks(data(cfg))
    te = df[m["test"]]
    sid, y = te.screen_id.to_numpy(), te.label.to_numpy()
    best, names = dev_names(cfg)
    (src_crop,) = crop_sources(cfg, "test")
    t0 = time.time()
    feats = embed(resnet18_trunk(cfg.get("pretrained", True)), np.load(cache_path(cfg, crop_size(cfg), "test"), mmap_mode="r"),
                  DEV, cfg["embed_batch"])
    print(f"frozen test features {time.time() - t0:.0f}s", flush=True)
    src_feat = C.Source("feat", feats, DEV)
    probs = {}
    for n, (_, ft) in names.items():
        probs[short(cfg, n, best)] = load_one(cfg, n, te, src_crop if ft else src_feat)
        print("scored", n, flush=True)
    np.savez_compressed(Path(cfg["ckpt_dir"]) / "element_types_cnn_test_probs.npz", screen_id=sid, y=y,
                        **{k: v.astype(np.float16) for k, v in probs.items()})
    probs = with_colour_model(cfg, "test", sid, y, probs)
    out = report(sid, y, probs, PAIRS, labels(cfg), ("model3_s0", "model2_s0"), cfg["n_boot"], cfg["seed"])
    out["model3_seeds"] = seeds_summary(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["features", "frozen", "cache", "tune", "dev", "test"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg["crop_workers"] = cfg.get("crop_workers") or os.cpu_count()
    run_dir = start_run(f"element_types_cnn_{args.stage}", {**cfg, "stage": args.stage}, root=cfg["runs_root"])
    print("run dir:", run_dir, "| device:", DEV, flush=True)
    t0 = time.time()
    res = {"features": stage_features, "frozen": stage_frozen, "cache": stage_cache, "tune": stage_tune,
           "dev": stage_dev, "test": stage_test}[args.stage](cfg, run_dir)
    res["stage_minutes"] = round((time.time() - t0) / 60, 1)
    finish_run(run_dir, res)
    print(json.dumps(res, indent=1, default=float)[:6000])
