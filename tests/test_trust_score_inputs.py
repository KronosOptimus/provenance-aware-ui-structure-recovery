"""Provenance checks: invalid nodes as elements, elements are leaves of the input tree, injected fakes never sit
under an element, and with no extra elements the provenance builder gives the Models 4-6 pool exactly."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uistruct.data import provenance_tokens as PT
from uistruct.data.features import FLAGS
from uistruct.data.nesting_tokens import build_screen

sys.path.insert(0, str(Path(__file__).parent))
from test_fusion_scorer import POOL, toy_screen  # noqa: E402

ROOT = Path(__file__).parents[1]


def toy_with_invalid():
    """toy_screen plus node 6: a CLAY-invalid node inside valid leaf 4 (a real raw-tree pattern)."""
    g, raw = toy_screen()
    raw = raw.assign(label=raw.node_idx.map({0: -1, 1: 5, 2: 1, 3: 2, 4: 3, 5: -1}), is_labelled=raw.node_idx != 5)
    extra = pd.DataFrame({"screen_id": 7, "node_idx": [6], "parent_idx": [4], "box_ok": [True],
                          "nx0": [.2], "ny0": [.62], "nx1": [.3], "ny1": [.68], "depth": [2], "n_children": [0],
                          "class_name": ["a.View"], **{f: [True] for f in FLAGS}, "label": [0], "is_labelled": [True]})
    raw = pd.concat([raw, extra[list(raw.columns)]], ignore_index=True).sort_values("node_idx").reset_index(drop=True)
    return g, raw


def test_invalid_element_under_a_leaf_is_built_as_a_leaf():
    g, raw = toy_with_invalid()
    vleaves = {2, 3, 4}
    assert PT.invalid_elements(raw, vleaves) == [6]
    sc = PT.build_level(g, raw, [6], POOL, 0.5, kinds={6: 1})
    assert list(sc["leaf_ids"]).count(6) == 1
    j = list(sc["leaf_ids"]).index(6)
    assert sc["leaf_trust"][j] == 0 and sc["leaf_kind"][j] == 1
    m = sc["M"]
    assert frozenset({4, 6}) not in m["sets"][:int(m["meta"][:, 0].sum())]   # leaf 4 is no raw container
    # candidate labels ignore the invalid member: {2, 3} stays the one positive, {4, 6} is not a container
    pos = {s - {6} for s, y in zip(m["sets"], m["y"]) if y}
    assert pos == {frozenset({2, 3})}
    assert all(not y for s, y in zip(m["sets"], m["y"]) if s <= {6})


def test_elements_as_leaves():
    chain = {2: ("raw", 1), 6: 4, 4: ("raw", 0), ("raw", 1): ("raw", 0), ("raw", 0): -1}
    out = PT.elements_as_leaves(chain, {2, 4, 6})
    assert out[6] == ("raw", 0) and out[2] == ("raw", 1) and out[4] == ("raw", 0)


def test_injected_fakes_never_sit_under_an_element():
    g, raw = toy_with_invalid()
    donors = {"screen_id": np.array([99] * 3), "box": np.array([[.2, .62, .3, .68], [.15, .13, .2, .2], [.6, .9, .7, .95]]),
              "class_name": np.array(["a.View"] * 3, object), "flags": np.ones((3, len(FLAGS)), bool),
              "n_children": np.zeros(3, int)}
    elements = {2, 3, 4, 6}
    r, fakes = PT.inject(raw, 30, donors, np.random.default_rng(0), elements)
    hosts = r.set_index("node_idx").loc[fakes, "parent_idx"]
    assert len(fakes) == 30 and not set(hosts) & elements


@pytest.mark.skipif(not (ROOT / "data/processed/elements.parquet").exists(), reason="processed tables only on the laptop")
def test_real_level_equals_structure_pool_without_extra_elements():
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_trust_score_on_corrupted_metadata as S
    cfg = S.load_cfg(str(ROOT / "configs/train_trust_score_on_corrupted_metadata.yaml"), True, False)
    cfg["data_dir"] = str(ROOT / cfg["data_dir"])
    sids, groups, raw_by, nodes = S.load_tables(cfg, "dev", sample=80, seed=3)
    donors = PT.donor_table(nodes)
    checked = 0
    for s in sids:
        g, raw = groups[s], raw_by[s]
        lv = PT.build_screen_levels(g, raw, cfg["pool"], cfg["min_iou"], ["real", "absent"], {"p_inj": 0.1}, donors, 11)["v"]
        old = build_screen(g, raw, cfg["pool"], cfg["min_iou"], pools=("P", "M"))
        assert lv["absent"]["M"]["sets"] == old["P"]["sets"]
        if (lv["real"]["leaf_kind"] > 0).any():
            continue
        a, b = lv["real"]["M"], old["M"]
        assert a["sets"] == b["sets"] and np.allclose(a["meta"], b["meta"]) and np.array_equal(a["y"], b["y"])
        assert np.array_equal(a["leaf_rawpar"], b["leaf_rawpar"]) and np.array_equal(lv["real"]["leaf_ids"], old["leaf_ids"])
        assert np.allclose(lv["real"]["leaf_meta"][:, :-1], old["leaf_meta"])
        checked += 1
    assert checked > 20


# ---------------------------------------------------------------- model, assembly, calibration
def test_trust_model_starts_as_model4_and_keeps_old_names():
    import torch
    from uistruct.models.fusion import FusionScorer
    from uistruct.train import provenance_head as PH
    g, raw = toy_with_invalid()
    sc = PT.build_level(g, raw, [6], POOL, 0.5, kinds={6: 1})
    enc = PH.ProvenanceEncoder(min_count=1).fit([sc])
    it = enc.transform(sc)
    b = PH.collate_fusion([it])
    kw = dict(d=16, heads=2, layers=2, ffn=32, dropout=0.0)
    torch.manual_seed(0)
    old = FusionScorer(enc.n_leaf_num, enc.n_cand_num, len(enc.names) + 2, len(enc.pieces) + 2, True, **kw).eval()
    new = FusionScorer(enc.n_leaf_num, enc.n_cand_num, len(enc.names) + 2, len(enc.pieces) + 2, True, trust=True, **kw).eval()
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert set(missing.missing_keys) == {k for k in new.state_dict() if k.startswith(("trust_head", "trust_bias"))}
    with torch.no_grad():
        a, c = old(b), new(b)
    assert torch.allclose(a[0], c[0], atol=1e-6)            # trust bias starts at 0: same candidate scores
    assert c[2][0].shape == c[2][1].shape == b["trust_tok"].shape
    ty, tt = PH.trust_arrays(sc)
    assert ty[list(sc["leaf_ids"]).index(6)] == 0 and tt[:len(sc["leaf_ids"])].all()


def test_absent_level_has_no_trust_claims():
    from uistruct.train import provenance_head as PH
    g, raw = toy_with_invalid()
    sc = PT.build_level(g, raw, [6], POOL, 0.5, absent=True)
    ty, tt = PH.trust_arrays(sc)
    assert not tt.any() and (ty == -1).all() and 6 not in sc["leaf_ids"]


def test_removal_tree():
    from uistruct.eval.metrics import VROOT, grouping_metrics
    from uistruct.train import provenance_head as PH
    g, raw = toy_with_invalid()
    sc = PT.build_level(g, raw, [6], POOL, 0.5, kinds={6: 1})
    scores = sc["M"]["y_iou"]
    rm_inv = np.array([l == 6 for l in sc["leaf_ids"]])
    par = PH.removal_tree(sc, scores, 0.5, rm_inv)
    assert 6 not in par and grouping_metrics(sc["true"], par)["container_f1"] == 1.0
    rm_valid = np.array([l == 2 for l in sc["leaf_ids"]])                # a valid leaf removed: under the root
    par = PH.removal_tree(sc, scores, 0.5, rm_valid)
    assert par[2] == VROOT and grouping_metrics(sc["true"], par)["container_f1"] < 1.0


def test_calibration_maps():
    from uistruct.eval import trust as TR
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-z))).astype(float)
    over = 1 / (1 + np.exp(-3 * z))                                      # over-confident by a factor 3
    t = TR.fit_temperature(over, y)
    assert abs(t["T"] - 3) < 0.2
    assert TR.ece(TR.apply_cal(t, over), y) < TR.ece(over, y)
    pl = TR.fit_platt(over, y)
    assert abs(pl["a"] - 1 / 3) < 0.03 and abs(pl["b"]) < 0.05
    s = TR.trust_scores(1 / (1 + np.exp(-z)), y, y.mean())
    assert s["brier_skill"] > 0 and 0.5 < s["auroc"] <= 1


def test_constant_column_not_blown_up():
    """A column constant in training (masked flag, real-only training) keeps std 1 at evaluation."""
    import numpy as np
    from uistruct.train.provenance_head import ProvenanceEncoder
    enc = ProvenanceEncoder.__new__(ProvenanceEncoder)
    enc.ls, enc.cs = np.array([1e-6, 2.0]), np.array([1e-6])
    enc.floor_std()
    assert enc.ls.tolist() == [1.0, 2.0] and enc.cs.tolist() == [1.0]
