import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from uistruct.data.features import FLAGS
from uistruct.data.nesting_tokens import build_screen, rule_trees
from uistruct.models import fusion
from uistruct.models.fusion import relation_matrix, raw_relation_matrix
from uistruct.models.nesting_raw import prune, raw_chain
from uistruct.train.nesting_head import TokenEncoder, build_model, collate, screen_tree, tree_rows

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
try:   # the nesting baselines script reads data/raw/clay/label_map.txt at import: present on the laptop, not on Colab
    import structure_rule_baselines as nb_script  # noqa: E402
except FileNotFoundError:
    nb_script = None

POOL = {"singletons": True, "all_leaves": True, "xy_tols": [0.0, 0.01], "agglo_methods": ["single"]}
HP = {"d": 16, "heads": 2, "layers": 2, "ffn": 32, "dropout": 0.0}


def toy_screen(shift=0.0):
    """raw: 0 root > 1 container (leaves 2, 3) and leaf 4; 5 is an unusable-box wrapper above leaf 3."""
    raw = pd.DataFrame({
        "screen_id": 7, "node_idx": [0, 1, 2, 5, 3, 4], "parent_idx": [-1, 0, 1, 1, 5, 0],
        "box_ok": [True, True, True, False, True, True],
        "nx0": [0, .05, .1, 0, .5, .1], "ny0": [0, .1 + shift, .12 + shift, 0, .12 + shift, .6],
        "nx1": [1, .95, .4, 0, .9, .9], "ny1": [1, .3 + shift, .28 + shift, 0, .28 + shift, .7],
        "depth": [0, 1, 2, 2, 3, 1], "n_children": [2, 2, 0, 1, 0, 0],
        "class_name": ["a.FrameLayout", "a.LinearLayout", "a.TextView", "a.View", "a.ImageView", "a.Button"],
        **{f: True for f in FLAGS}}).sort_values("node_idx")
    g = raw[raw.node_idx.isin([1, 2, 3, 4])].assign(gt_parent=[-1, 1, 1, -1], label=[5, 1, 2, 3])
    g = g.assign(gt_parent=g.node_idx.map({1: -1, 2: 1, 3: 1, 4: -1}), label=g.node_idx.map({1: 5, 2: 1, 3: 2, 4: 3}))
    return g, raw


@pytest.mark.skipif(nb_script is None, reason="raw CLAY files are only on the laptop")
def test_raw_helpers_match_baselines_script():
    g, raw = toy_screen()
    leaves = {2, 3, 4}
    assert raw_chain(raw, leaves) == nb_script.raw_chain(raw, leaves)
    assert prune(raw_chain(raw, leaves), leaves) == nb_script.prune(nb_script.raw_chain(raw, leaves), leaves)


def test_build_screen_labels_and_pools():
    g, raw = toy_screen()
    sc = build_screen(g, raw, POOL)
    for k in "PM":
        sets = sc[k]["sets"]
        pos = [s for s, y in zip(sets, sc[k]["y"]) if y]
        assert pos == [frozenset({2, 3})]                     # the one true container
    m = sc["M"]
    assert m["meta"][:, 0].sum() == 2                         # raw candidates: root and node 1 (5 has no box)
    assert "meta" not in sc["P"] and "name" not in sc["P"]    # pixels-only pool carries no metadata
    ok = sc["true"]
    pred = screen_tree(sc, "P", sc["P"]["y_iou"], 0.5)
    assert tree_rows(sc, pred, "x")[0]["container_f1"] == 1.0 and ok[2] == 1
    assert set(rule_trees(g, raw, 0.002)) == {"xycut", "raw_chain_collapsed"}


def test_relation_matrix():
    member = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]], bool)
    R = relation_matrix(member)
    L = 3
    assert R[0, L + 0] == fusion.REL_L_IN and R[2, L + 0] == fusion.REL_L_OUT
    assert R[L + 1, L + 0] == fusion.REL_SUB and R[L + 0, L + 1] == fusion.REL_SUPER
    assert R[L + 0, L + 3] == fusion.REL_EQUAL and R[L + 0, L + 2] == fusion.REL_CROSS
    assert R[L + 1, L + 4] == fusion.REL_APART and (np.diag(R) == fusion.REL_SELF).all()
    rr = raw_relation_matrix(np.array([0, 0, -1]), np.array([-1, 0]))
    assert rr[0, 3] == 1 and rr[3, 0] == 2 and rr[3 + 1, 3] == 1


def _items(use_meta, pool):
    screens = [build_screen(*toy_screen(s), POOL) for s in (0.0, 0.3)]
    enc = TokenEncoder(pool, use_meta, min_count=1).fit(screens)
    return enc, [enc.transform(s) for s in screens]


def test_padding_and_order_do_not_change_scores():
    for use_meta, pool in [(True, "M"), (False, "P")]:
        enc, items = _items(use_meta, pool)
        torch.manual_seed(0)
        model = build_model(HP, enc).eval()
        big = dict(items[1])
        with torch.no_grad():
            alone = model(collate([items[0]]))[0][0, :len(items[0]["y"])]
            padded = model(collate([items[0], big]))[0][0, :len(items[0]["y"])]
            assert torch.allclose(alone, padded, atol=1e-5)
            # reverse the leaf order (and the membership / relation matrices with it)
            it = dict(items[0])
            L = len(it["leaf_num"])
            perm = np.arange(L)[::-1].copy()
            tok = np.r_[perm, L:L + len(it["cand_num"])]
            for k in ["leaf_num", "leaf_label", "leaf_name", "leaf_pieces"]:
                if k in it:
                    it[k] = it[k][perm]
            it["member"] = it["member"][:, perm]
            it["rel"] = it["rel"][np.ix_(tok, tok)]
            if "raw_rel" in it:
                it["raw_rel"] = it["raw_rel"][np.ix_(tok, tok)]
            flipped = model(collate([it]))[0][0, :len(it["y"])]
            assert torch.allclose(alone, flipped, atol=1e-5)


def test_pixels_only_model_has_no_metadata_path():
    enc, items = _items(False, "P")
    model = build_model(HP, enc)
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.startswith(("name_emb", "piece_emb", "raw_bias")) for n in names)
    assert "leaf_name" not in items[0] and "raw_rel" not in items[0] and enc.n_leaf_num == 6
