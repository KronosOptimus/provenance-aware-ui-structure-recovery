import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from uistruct.data.splits import clean_splits
from uistruct.data.tree import NO_PARENT, VROOT, build_elements
from uistruct.eval.metrics import (calibration, density_bin, grouping_metrics, is_forest, legacy_macro_f1,
                                   smooth_single_child, tree_metrics, type_metrics)
from uistruct.eval.stats import bootstrap_mean, paired_bootstrap

TRUE = {1: -1, 2: 1, 3: 1, 4: -1, 5: 4}


def test_tree_metrics_hand_computed():
    pred = {1: -1, 2: 1, 3: -1, 4: -1, 5: 4}  # node 3 moved to the top level
    m = tree_metrics(TRUE, pred)
    assert m["edge_f1"] == pytest.approx(0.8)          # 4 of 5 edges right
    assert m["leaf_edge_f1"] == pytest.approx(2 / 3)   # leaves 2,3,5: 2 of 3 edges right
    assert m["container_match"] == pytest.approx(0.75)  # container 1: {2,3} vs {2} = 0.5; container 4: 1.0
    assert m["edit_distance"] == 2                    # delete one edge, insert one


def test_perfect_and_flat():
    assert tree_metrics(TRUE, TRUE)["edge_f1"] == 1 and tree_metrics(TRUE, TRUE)["container_match"] == 1
    flat = {k: -1 for k in TRUE}
    m = tree_metrics(TRUE, flat)
    assert m["container_match"] == 0 and m["edge_f1"] == pytest.approx(0.4)


def test_container_match_ignores_container_identity():
    # same grouping {2,3}, but a different node is named as the container
    true = {1: -1, 6: -1, 2: 1, 3: 1}
    pred = {1: -1, 6: -1, 2: 6, 3: 6}
    m = tree_metrics(true, pred)
    assert m["container_match"] == 1.0 and m["edge_f1"] == 0.5


def test_no_true_container_is_nan():
    assert np.isnan(tree_metrics({1: -1, 2: -1}, {1: -1, 2: -1})["container_match"])


def test_grouping_metrics_created_containers():
    # true: container 1 over leaves {2,3}, leaf 4 at top level
    true = {1: -1, 2: 1, 3: 1, 4: -1}
    same = {"a": -1, 2: "a", 3: "a", 4: -1}           # same grouping, new container id
    m = grouping_metrics(true, same)
    assert m["container_match"] == 1 and m["edge_f1"] == 1 and m["edit_distance"] == 0
    wrong = {"a": -1, 2: "a", 4: "a", 3: -1}           # groups {2,4}: IoU 1/3 < 0.5, unmatched
    m = grouping_metrics(true, wrong)
    assert m["container_match"] == pytest.approx(1 / 3)
    assert m["edge_f1"] == pytest.approx(0.0)          # no edge survives: container is unmatched
    flat = {2: -1, 3: -1, 4: -1}
    m = grouping_metrics(true, flat)
    assert m["container_match"] == 0 and m["leaf_edge_f1"] == pytest.approx(1 / 3)  # only (4, root) of 3 leaf edges right
    assert np.isnan(m["container_precision"]) and m["container_f1"] == 0


def test_container_precision_punishes_extra_containers():
    true = {1: -1, 2: 1, 3: 1, 4: -1, 5: -1}
    # right group {2,3} plus a fake wrapper over all leaves: CM stays 1, but only 1 of 2 predicted containers is real
    over = {"w": -1, "a": "w", 2: "a", 3: "a", 4: "w", 5: "w"}
    m = grouping_metrics(true, over)
    assert m["container_match"] == 1
    assert m["container_precision"] == pytest.approx(0.5)
    assert m["container_f1"] == pytest.approx(2 / 3)    # 2 * 1 / (1 + 2)


def test_grouping_metrics_nested_same_leaf_set():
    # containers 1 and 2 cover the same leaves {3,4}; the identity prediction must score perfectly
    true = {1: -1, 2: 1, 3: 2, 4: 2, 5: -1}
    m = grouping_metrics(true, dict(true))
    assert m["edge_f1"] == 1 and m["container_match"] == 1
    assert grouping_metrics(true, true)["edge_f1"] == tree_metrics(true, true)["edge_f1"]


def test_smooth_single_child():
    assert smooth_single_child({1: -1, 2: 1, 5: 2, 6: 2}) == {2: -1, 5: 2, 6: 2}
    assert smooth_single_child({1: -1, 2: 1, 3: 2}) == {3: -1}
    # a kept node (given leaf with one child) stays; the wrapper above it still collapses
    assert smooth_single_child({1: -1, 2: 1, 3: 2}, keep={2}) == {2: -1, 3: 2}


def test_forest_check():
    assert is_forest(TRUE)
    assert not is_forest({1: 2, 2: 1})
    assert not is_forest({1: 9})
    with pytest.raises(ValueError):
        tree_metrics(TRUE, {1: 2, 2: 1, 3: -1, 4: -1, 5: 4})


def test_type_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 24, 2000)
    p = np.where(rng.random(2000) < 0.6, t, rng.integers(0, 24, 2000))
    m = type_metrics(t, p)
    assert m["macro_f1_types"] == pytest.approx(f1_score(t, p, labels=list(range(1, 24)), average="macro"))
    assert m["invalid_f1"] == pytest.approx(f1_score(t == 0, p == 0))
    t2 = np.r_[t, -1, -1]
    p2 = np.r_[p, -1, 3]
    assert legacy_macro_f1(t2, p2) == pytest.approx(f1_score(t2, p2, labels=list(range(-1, 24)), average="macro"))


def test_calibration():
    c = calibration([0.9] * 10 + [0.1] * 10, [1] * 9 + [0] + [0] * 9 + [1])
    assert c["ece"] == pytest.approx(0.0) and c["auroc"] > 0.5
    c = calibration([1.0] * 4, [1, 1, 0, 0])
    assert c["ece"] == pytest.approx(0.5)


def test_density_bins():
    assert [density_bin(n) for n in (0, 15, 16, 63, 64, 300)] == ["0-16", "0-16", "16-32", "48-64", "64+", "64+"]


def test_bootstrap():
    b = bootstrap_mean(np.r_[np.ones(50), np.zeros(50), np.nan])
    assert b["mean"] == 0.5 and b["lo"] < 0.5 < b["hi"] and b["n"] == 100
    p = paired_bootstrap(np.ones(30), np.zeros(30))
    assert p["diff"] == 1 and p["p_a_not_better"] == 0


def test_build_elements():
    # ROOT(0) -> BG(1) -> TEXT(2);  ROOT -> CONTAINER(3) -> TEXT(4);  unlabelled(5) under 3
    n = pd.DataFrame({"screen_id": 1, "node_idx": range(6), "clay_parent_idx": [-1, 0, 1, 0, 3, 3],
                      "label": [-1, 0, 4, 21, 4, -2], "is_labelled": [True] * 5 + [False]})
    e = build_elements(n).set_index("node_idx")
    assert list(e.index) == [1, 2, 3, 4]
    assert e.gt_parent.to_dict() == {1: NO_PARENT, 2: VROOT, 3: VROOT, 4: 3}
    assert e.is_valid.tolist() == [False, True, True, True]


def test_clean_splits():
    s = pd.DataFrame({"screen_id": range(7), "split": ["train"] * 4 + ["dev", "dev", "test"],
                      "activity_name": ["a.x.p/A", "b.y.p/A", "c.z.p/A", "d.d.p/A", "b.y.q/A", "c.z.r/A", "c.z.s/A"]})
    h = pd.Series({0: "h0", 1: "h1", 2: "h2", 3: "h6", 4: "h4", 5: "h5", 6: "h6"})
    out = clean_splits(s, h).set_index("screen_id").split_clean
    # 1: company b.y in dev; 2: company c.z in test; 3: same layout as a test screen; 5: dev company in test
    assert out.tolist() == ["train", "dropped", "dropped", "dropped", "dev", "dropped", "test"]
    assert clean_splits(s, h, drop_dev_test_overlap=False).set_index("screen_id").split_clean[5] == "dev"
