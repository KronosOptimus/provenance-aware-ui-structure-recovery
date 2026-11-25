"""Demo pipeline: one held-out test screen with its metadata at a chosen level in, a labelled tree out,
with a trust score per element and, per container, whether it was read from the metadata or inferred.

It runs the planned provenance system: model fusion_full_s0, elements whose trust is below t_e are
removed, then the tree is assembled from candidates scoring at least t_c. t_c and t_e are the validation
choices stored in the test run (results/runs/colab_trust_score/trust_score_test_run1). Screens, candidate pool and
corruption are built exactly as scripts/train_trust_score_on_corrupted_metadata.py builds the test split: same seed
(corruption_seed + 2), and fakes are copied from the invalid elements of every test screen.
Differences from the test run:
  - crop embeddings are computed on the CPU in float32, then stored as float16 (Colab: float16 autocast);
  - no element is removed at the absent level (plan section 7). The test-run code removed 0.08 elements
    per screen there, so at that level the demo matches the "nothing removed" row.
CLAY labels are used only to build the element list (as in training) and the answer key, never as model inputs.
Trust scores are the raw head outputs, which is what the removal threshold was tuned on.
"""
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from uistruct.data import provenance_tokens as PT
from uistruct.data.crops import screen_crops
from uistruct.data.features import FLAGS
from uistruct.data.nesting_crops import SIZE, screen_boxes
from uistruct.eval.metrics import VROOT, grouping_metrics
from uistruct.models.cnn import Normalise, resnet18_trunk
from uistruct.models.nesting import leaf_sets_of
from uistruct.train import provenance_head as PH

BOX = PT.BOX
RAW_COLS = ["screen_id", "node_idx", "parent_idx", "box_ok", *BOX, "depth", "n_children", "class_name", *FLAGS]
LEVELS = ["real", "heavy", "inject2", "drop50", "absent"]
LEVEL_TEXT = {"real": "Real metadata (as the app delivered it)",
              "heavy": "Heavy corruption (fakes added, 25% of nodes dropped, 25% hidden)",
              "inject2": "Twice the real error rate (fake elements added)",
              "drop50": "Half the metadata missing (50% of nodes dropped, 50% hidden)",
              "absent": "No metadata (pixels and element boxes only)"}
KIND_TEXT = {0: "valid", 1: "real metadata error (CLAY invalid)", 2: "injected fake"}
SPLIT_SEED_TEST = 2           # scripts/train_trust_score_on_corrupted_metadata.py SPLIT_SEED["test"]
CROP_DIM = 512
SYSTEM = "fusion_full_s0+trust"


def label_names(path="data/raw/clay/label_map.txt"):
    out = {}
    for line in open(path, encoding="utf-8"):
        if ":" in line:
            k, v = line.split(":", 1)
            out[int(k)] = v.strip()
    return out


def load_test_tables(data_dir, sids=None):
    """As scripts/train_trust_score_on_corrupted_metadata.py load_tables for the test split: valid elements by screen, raw nodes by
    screen (CLAY label attached for truth only) and the raw node table."""
    el_all = pd.read_parquet(f"{data_dir}/elements.parquet",
                             columns=["screen_id", "node_idx", "gt_parent", "label", "is_valid", "split_clean", *BOX])
    el_all = el_all[el_all.split_clean == "test"]
    el = el_all[el_all.is_valid]
    all_sids = sorted(el.screen_id.unique())
    sids = all_sids if sids is None else sorted(set(sids) & set(all_sids))
    el = el[el.screen_id.isin(set(sids))]
    nodes = pd.read_parquet(f"{data_dir}/nodes.parquet", columns=RAW_COLS, filters=[("screen_id", "in", sids)])
    nodes = nodes.merge(el_all[["screen_id", "node_idx", "label"]], on=["screen_id", "node_idx"], how="left",
                        validate="one_to_one")
    nodes["is_labelled"] = nodes.label.notna()
    nodes["label"] = nodes.label.fillna(-1).astype(np.int8)
    raw_by = {k: v.sort_values("node_idx").reset_index(drop=True) for k, v in nodes.groupby("screen_id", sort=False)}
    groups = dict(tuple(el.groupby("screen_id", sort=True)))
    return sids, groups, raw_by, nodes


def _invalid_one(args):
    g, raw = args
    true = dict(zip(g.node_idx.tolist(), g.gt_parent.tolist()))
    return PT.invalid_elements(raw, set(true) - set(true.values()))


def test_donors(data_dir, cache, workers=6, log=print):
    """Fake donors = the invalid elements of every test screen (as split_donors in scripts/train_trust_score_on_corrupted_metadata.py); cached."""
    cache = Path(cache)
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    t0 = time.time()
    sids, groups, raw_by, nodes = load_test_tables(data_dir)
    with ProcessPoolExecutor(workers) as ex:
        inv = dict(zip(sids, ex.map(_invalid_one, [(groups[s], raw_by[s]) for s in sids], chunksize=64)))
    d = PT.donor_table(nodes, inv)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(d, open(cache, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log(f"donors: {len(d['screen_id'])} invalid elements from {len(sids)} test screens ({time.time() - t0:.0f}s)")
    return d


def cf1_rule(m):
    """Empty-screen rule: a screen with no true and no predicted container scores 1.0."""
    if m["n_true_containers"] == 0 and m["n_pred_containers"] == 0:
        return 1.0
    return m["container_f1"]


class Demo:
    def __init__(self, root=".", model_dir="data/models/fusion_full_s0", model3_ckpt="data/models/element_types_cnn/model3_lr2.pt",
                 config="configs/train_trust_score_on_corrupted_metadata.yaml", test_run="results/runs/colab_trust_score/trust_score_test_run1",
                 donor_cache="data/processed/demo/donors_test.pkl", workers=6, log=print):
        root = Path(root)
        cfg = yaml.safe_load(open(root / config))
        cfg.update(cfg["laptop"])
        self.cfg, self.root, self.log = cfg, root, log
        self.data_dir, self.img_dir = root / cfg["data_dir"], root / cfg["img_dir"]
        self.rates = {"p_inj": json.load(open(root / cfg["rates_run"]))["train"]["p_inj"]}
        self.seed = cfg["corruption_seed"] + SPLIT_SEED_TEST
        tuned = json.load(open(root / test_run / "metrics.json"))["tuned"][SYSTEM]
        self.t_c, self.t_e = float(tuned["t_c"]), float(tuned["t_e"])
        st = torch.load(root / model_dir / "model.pt", map_location="cpu", weights_only=False)
        assert st["variant"] == "fusion_full" and st["seed"] == 0 and st["trust"], "not the planned provenance model"
        self.hp = st["hp"]
        self.enc = pickle.load(open(root / model_dir / "enc.pkl", "rb"))
        self.model = PH.build_fusion(self.hp, self.enc, CROP_DIM, True)
        self.model.load_state_dict(st["state"])
        self.model.eval()
        ck = torch.load(root / model3_ckpt, map_location="cpu", weights_only=False)
        assert ck["fine_tune"] and abs(ck["hp"]["lr_backbone"] - cfg["model3_lr_backbone"]) < 1e-12, "not Model 3"
        self.trunk = resnet18_trunk(pretrained=False)
        self.trunk.load_state_dict({k[len("backbone."):]: v for k, v in ck["state"].items() if k.startswith("backbone.")})
        self.trunk.eval()
        self.norm = Normalise()
        self.donors = test_donors(self.data_dir, root / donor_cache, workers, log)
        self.names = label_names(root / "data/raw/clay/label_map.txt")
        self.groups, self.raw = {}, {}

    # ---------------------------------------------------------------- inputs
    def load(self, sids):
        """Read the tables of these test screens once (a few seconds per call)."""
        need = [s for s in sids if s not in self.groups]
        if need:
            got, groups, raw_by, _ = load_test_tables(self.data_dir, need)
            for s in got:
                self.groups[s], self.raw[s] = groups[s], raw_by[s]
        return [s for s in sids if s in self.groups]

    def level(self, sid, lv):
        """One level of one screen, identical to the test cache (the seed is per screen and level)."""
        self.load([sid])
        return PT.build_screen_levels(self.groups[sid], self.raw[sid], self.cfg["pool"], self.cfg["min_iou"], [lv],
                                      self.rates, self.donors, self.seed)["v"][lv]

    @torch.no_grad()
    def crops(self, sid, v):
        uniq, idx = screen_boxes(None, {"leaf": v["leaf_xyxy"], "M": v["M"]["crop_xyxy"]})
        c = screen_crops(f"{self.img_dir}/{sid}.jpg", uniq, [SIZE])[SIZE]
        e = self.trunk(self.norm(torch.from_numpy(c))).numpy().astype(np.float16)
        return e[idx["leaf"]], e[idx["M"]]

    # ---------------------------------------------------------------- one run
    @torch.no_grad()
    def predict(self, sid, v):
        item = self.enc.transform(v, self.crops(sid, v))
        c, t, r = PH.predict_fusion(self.model, [item], self.hp["budget"])
        L = len(v["leaf_ids"])
        return c[0], t[0], r[0][:L]

    def run(self, sid, lv):
        t0 = time.time()
        v = self.level(sid, lv)
        cand, typ, trust = self.predict(sid, v)
        has_meta = v["has_meta"]
        removed = (trust < self.t_e) if has_meta else np.zeros(len(v["leaf_ids"]), bool)
        par = PH.removal_tree(v, cand, self.t_c, removed)
        m = grouping_metrics(v["true"], par)
        return self._result(sid, lv, v, cand, typ, trust, removed, par, m, time.time() - t0)

    # ---------------------------------------------------------------- output for the app
    def _result(self, sid, lv, v, cand, typ, trust, removed, par, m, sec):
        lids = v["leaf_ids"].tolist()
        pos = {l: i for i, l in enumerate(lids)}
        nr = int((v["M"]["meta"][:, 0] > 0).sum())
        rem = {l for l, r in zip(lids, removed) if r}
        box = v["leaf_xyxy"]
        conts = [k for k in par if isinstance(k, tuple) and k[0] == "c"]

        def depth(n):
            d = 0
            while par.get(n, VROOT) != VROOT:
                n, d = par[n], d + 1
            return d
        name = lambda k: ("c%d" % k[1]) if isinstance(k, tuple) else ("e%d" % k)  # noqa: E731
        members = {k: sorted(v["M"]["sets"][k[1]] - rem, key=pos.get) for k in conts}
        containers = []
        for k in sorted(conts, key=lambda k: (depth(k), -len(members[k]))):
            b = box[[pos[l] for l in members[k]]]
            j = k[1]
            containers.append({"id": name(k), "parent": None if par[k] == VROOT else name(par[k]), "depth": depth(k),
                               "box": [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())],
                               "members": [name(l) for l in members[k]], "score": round(float(cand[j]), 3),
                               "source": "metadata" if j < nr else "inferred",
                               "class_name": v["M"]["name"][j] if j < nr else ""})
        elements = []
        for i, l in enumerate(lids):
            p = typ[i]
            t = int(np.argmax(p[1:]) + 1)
            elements.append({"id": name(l), "parent": None if l in rem or par.get(l, VROOT) == VROOT else name(par[l]),
                             "box": [float(x) for x in box[i]], "type": self.names.get(t, str(t)),
                             "type_p": round(float(p[t]), 3), "p_invalid": round(float(p[0]), 3),
                             "trust": round(float(trust[i]), 3) if v["has_meta"] else None,
                             "removed": bool(removed[i]), "masked": bool(v["leaf_meta"][i, -1] > 0),
                             "class_name": v["leaf_name"][i], "kind": KIND_TEXT[int(v["leaf_kind"][i])]})
        vleaves = set(v["true"]) - set(v["true"].values())
        truth = []
        for _, s in leaf_sets_of(v["true"], vleaves):
            b = box[[pos[l] for l in s if l in pos]]
            truth.append({"box": [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())],
                          "members": sorted(name(l) for l in s)})
        kind = v["leaf_kind"]
        n_read = sum(c["source"] == "metadata" for c in containers)
        return {"screen_id": int(sid), "level": lv, "level_text": LEVEL_TEXT[lv], "has_meta": bool(v["has_meta"]),
                "t_c": self.t_c, "t_e": self.t_e if v["has_meta"] else None, "seconds": round(sec, 2),
                "elements": elements, "containers": containers, "truth": truth,
                "summary": {"elements": len(lids), "removed": int(removed.sum()), "containers": len(containers),
                            "read_from_metadata": n_read, "inferred": len(containers) - n_read,
                            "masked": int((v["leaf_meta"][:, -1] > 0).sum())},
                "answer": {"container_f1": round(cf1_rule(m), 3), "true_containers": int(m["n_true_containers"]),
                           "unfaithful_elements": int((kind > 0).sum()),
                           "removed_unfaithful": int((removed & (kind > 0)).sum()),
                           "removed_valid": int((removed & (kind == 0)).sum()),
                           "kept_unfaithful": int((~removed & (kind > 0)).sum())}}
