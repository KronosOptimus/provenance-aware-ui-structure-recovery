"""Models 4-6 checks: crop regions (no raw box in the pixels-only pool), box de-duplication, crop inputs through the
scorer, equal tuning draws, the backbone rule and the trees comparator with crop PCA."""
import sys
from pathlib import Path

import numpy as np
import torch

from uistruct.data import nesting_crops as NC
from uistruct.data.nesting_tokens import build_screen
from uistruct.train.nesting_head import TokenEncoder, build_model, collate, fit_gbdt, gbdt_predict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from test_fusion_scorer import HP, POOL, toy_screen  # noqa: E402
import train_structure_models as S  # noqa: E402


def union_of_members(sc, k):
    lb = sc["leaf_xyxy"]
    return np.array([[lb[m, 0].min(), lb[m, 1].min(), lb[m, 2].max(), lb[m, 3].max()] for m in sc[k]["member"]])


def test_crop_regions():
    g, raw = toy_screen()
    sc = build_screen(g, raw, POOL)
    # pixels-only pool: every crop region is the box around the member leaves, never a raw box
    assert np.allclose(sc["P"]["crop_xyxy"], union_of_members(sc, "P"))
    m = sc["M"]
    nr = int(m["meta"][:, 0].sum())
    rw = raw.set_index("node_idx")
    rawboxes = rw.loc[[1, 0], ["nx0", "ny0", "nx1", "ny1"]].to_numpy(float)   # raw candidates: node 1 and root
    assert {tuple(np.round(b, 6)) for b in m["crop_xyxy"][:nr]} == {tuple(np.round(b, 6)) for b in rawboxes}
    assert np.allclose(m["crop_xyxy"][nr:], union_of_members(sc, "M")[nr:])   # generated ones: union


def test_screen_boxes_dedup():
    g, raw = toy_screen()
    sc = build_screen(g, raw, POOL)
    u, idx = NC.screen_boxes(sc)
    assert len(u) == len(np.unique(np.round(u, 6), axis=0))
    for k, boxes in [("leaf", sc["leaf_xyxy"]), ("P", sc["P"]["crop_xyxy"]), ("M", sc["M"]["crop_xyxy"])]:
        assert np.allclose(u[idx[k]], boxes, atol=1e-6)
    # a singleton candidate shares its leaf's box
    single = [i for i, mm in enumerate(sc["P"]["member"]) if mm.sum() == 1]
    assert single and all(idx["P"][i] in set(idx["leaf"]) for i in single)


def fake_crops(sc, pool, rng):
    return (rng.normal(size=(len(sc["leaf_ids"]), 512)).astype(np.float16),
            rng.normal(size=(len(sc[pool]["sets"]), 512)).astype(np.float16))


def test_scorer_with_crops_and_leak_free_pixels_only():
    rng = np.random.default_rng(0)
    scs = [build_screen(*toy_screen(s), POOL) for s in (0.0, 0.1)]
    enc = TokenEncoder("P", False).fit(scs)
    crops = [fake_crops(s, "P", rng) for s in scs]
    enc.fit_crops([c[0] for c in crops], [c[1] for c in crops])
    items = [enc.transform(s, c) for s, c in zip(scs, crops)]
    assert items[0]["cand_crop"].dtype == np.float16 and "cand_name" not in items[0]
    model = build_model(HP, enc, crop_dim=512)
    lc, lt = model(collate(items))
    assert lc.shape[0] == 2 and torch.isfinite(lc).all()
    names = [n for n, _ in model.named_parameters()]
    assert any(n.startswith("crop_proj") for n in names)
    assert not any(n.startswith(("name_emb", "piece_emb", "raw_bias")) for n in names)


def test_trial_draws_equal_across_models_and_trial0_is_config():
    base = {"d": 128, "ffn": 256, "lr": 1e-3, "layers": 3}
    tune = {"trials": 4, "seed": 0, "space": {"lr": {"log": [3e-4, 3e-3]}, "d": [64, 128, 256], "layers": [2, 3]}}
    a, b = S.trial_settings(base, tune), S.trial_settings(base, tune)
    assert a == b and a[0] == base and len(a) == 4
    assert all(3e-4 <= t["lr"] <= 3e-3 and t["ffn"] == 2 * t["d"] for t in a[1:])


def test_backbone_rule():
    cfg = {"backbone_tie": 0.002}
    assert S.pick_backbone(cfg, {"model3": {"dev_ap": 0.801}, "imagenet": {"dev_ap": 0.800}}) == "imagenet"
    assert S.pick_backbone(cfg, {"model3": {"dev_ap": 0.810}, "imagenet": {"dev_ap": 0.800}}) == "model3"
    assert S.pick_backbone(cfg, {"model3": {"dev_ap": 0.790}, "imagenet": {"dev_ap": 0.800}}) == "imagenet"


def test_gbdt_with_crop_pca():
    rng = np.random.default_rng(0)
    scs = [build_screen(*toy_screen(s), POOL) for s in np.linspace(0, 0.2, 12)]
    enc = TokenEncoder("M", True).fit(scs)
    crops = [fake_crops(s, "M", rng) for s in scs]
    enc.fit_crops([c[0] for c in crops], [c[1] for c in crops])
    hp = {"learning_rate": 0.1, "max_leaf_nodes": 7, "l2": 1.0, "min_samples_leaf": 2, "max_iter": 5, "max_cats": 10}
    clf = fit_gbdt(hp, scs, "M", True, enc, 0, crops, pca_dim=4)
    p = gbdt_predict(clf, scs, "M", True, enc, crops)
    assert len(p) == len(scs) and all(len(x) == len(s["M"]["sets"]) for x, s in zip(p, scs))
