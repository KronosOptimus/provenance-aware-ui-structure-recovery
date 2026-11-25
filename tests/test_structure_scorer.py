import numpy as np

from uistruct.eval.metrics import VROOT, grouping_metrics, is_forest
from uistruct.models.nesting import (agglomerative_sets, assemble, generated_candidates, leaf_sets_of,
                                     oracle_scores)

# true tree: root container 10 holds container 11 (leaves 1, 2) and wrapper 12 -> 13 -> leaf 3
TRUE = {10: VROOT, 11: 10, 1: 11, 2: 11, 12: 10, 13: 12, 3: 13, 4: VROOT}
LEAVES = {1, 2, 3, 4}


def test_assemble_keeps_valid_tree_and_drops_crossing():
    cands = [("a", frozenset({1, 2}), 0.9), ("b", frozenset({2, 3}), 0.8), ("c", frozenset({1, 2, 3}), 0.7)]
    par = assemble(cands, LEAVES, 0.5)
    assert is_forest(par)
    assert "b" not in par                       # crosses a
    assert par["a"] == "c" and par[1] == "a" and par[3] == "c" and par[4] == VROOT


def test_assemble_allows_stacks():
    cands = [("x", frozenset({3}), 0.9), ("y", frozenset({3}), 0.6)]
    par = assemble(cands, {3}, 0.5)
    assert par["x"] == VROOT and par["y"] == "x" and par[3] == "y"


def test_oracle_on_the_true_sets_is_perfect():
    pool = leaf_sets_of(TRUE, LEAVES)
    sets = [s for _, s in pool]
    sc = oracle_scores([s for _, s in leaf_sets_of(TRUE, LEAVES)], sets)
    pred = assemble([(("c", i), s, x) for i, ((_, s), x) in enumerate(zip(pool, sc))], LEAVES, 0.5)
    m = grouping_metrics(TRUE, pred)
    assert m["container_f1"] == 1.0 and m["edge_f1"] == 1.0


def test_generated_candidates_from_boxes():
    b = np.array([[0, 0, .4, .1], [.5, 0, .9, .1], [0, .5, .9, .6]])
    gen = generated_candidates([1, 2, 3], b, {"xy_tols": [0.0], "agglo_methods": ["single"]})
    assert frozenset({1, 2}) in gen and frozenset({1, 2, 3}) in gen and frozenset({3}) in gen
    assert agglomerative_sets([1], b[:1], "single") == []
