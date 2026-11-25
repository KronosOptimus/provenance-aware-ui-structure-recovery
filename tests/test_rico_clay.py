from uistruct.data.rico_clay import UNLABELLED, add_derived, parse_screen
import pandas as pd


def node(ptr, bounds, children=(), **kw):
    return {"pointer": ptr, "bounds": bounds, "children": list(children), "class": "V", **kw}


# root(R) -> wrapper(w, unlabelled) -> [btn(b), txt(t)] ; root -> img(i) ; plus a repeated pointer b
TREE = node("r", [0, 0, 1440, 2560], [
    node("w", [0, 0, 1440, 1000], [
        node("b", [0, 0, 720, 100], clickable=True),
        node("t", [720, 0, 1440, 100], text="hi", **{"content-desc": [None]}),
    ]),
    None,  # RICO sometimes has null children
    node("i", [0, 1000, 1440, 2560], [node("b", [0, 0, 0, 0])]),
])
LABELS = {"r": -1, "b": 3, "t": 4, "i": 1}


def rows():
    return parse_screen(7, TREE, LABELS)


def test_preorder_and_parents():
    r = rows()
    assert [x["pointer"] for x in r] == ["r", "w", "b", "t", "i", "b"]
    assert [x["parent_idx"] for x in r] == [-1, 0, 1, 1, 0, 4]
    assert [x["depth"] for x in r] == [0, 1, 2, 2, 1, 2]
    assert r[0]["n_children"] == 2  # the null child is dropped


def test_clay_parent_skips_unlabelled_wrapper():
    r = rows()
    assert r[2]["clay_parent_idx"] == 0  # b's raw parent is w (unlabelled) -> nearest labelled is r
    assert r[3]["clay_parent_idx"] == 0
    assert r[4]["clay_parent_idx"] == 0
    assert r[0]["clay_parent_idx"] == -1


def test_duplicate_pointer_first_copy_keeps_label():
    r = rows()
    assert r[2]["label"] == 3 and not r[2]["dup_pointer"]
    assert r[5]["label"] == UNLABELLED and r[5]["dup_pointer"]
    assert r[1]["label"] == UNLABELLED  # wrapper not labelled by CLAY


def test_flags_text_and_derived():
    r = rows()
    assert r[2]["clickable"] and not r[3]["clickable"]
    assert r[3]["text"] == "hi" and r[3]["content_desc"] is None
    df = add_derived(pd.DataFrame(r))
    assert df.box_ok.tolist() == [True, True, True, True, True, False]
    assert df.loc[3, "nx0"] == 0.5 and df.loc[4, "ny0"] == 1000 / 2560
    assert df.is_labelled.sum() == 4
