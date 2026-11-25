"""RICO view hierarchy + CLAY labels -> one flat node table.

One row per RICO node, in pre-order. Nodes are keyed by (screen_id, node_idx), not by the RICO
pointer, because a pointer can repeat inside one screen: the first copy
in pre-order keeps the CLAY label and later copies are left unlabelled with `dup_pointer=True`.

Coordinates: RICO bounds are in a 1440 x 2560 frame. Normalised boxes divide by that frame, so they
are independent of the screenshot resolution (all screenshots are 9:16; checked in build_node_table.py).

The column set is platform-neutral so that a web DOM adapter can emit the same table later.
"""
import json
from pathlib import Path

import pandas as pd

FRAME_W, FRAME_H = 1440, 2560
UNLABELLED = -2  # CLAY labels run -1 (ROOT) .. 23; -2 marks a node CLAY did not label

FLAGS = {  # output column -> RICO key
    "clickable": "clickable",
    "long_clickable": "long-clickable",
    "focusable": "focusable",
    "scrollable_v": "scrollable-vertical",
    "scrollable_h": "scrollable-horizontal",
    "enabled": "enabled",
    "selected": "selected",
    "visible_to_user": "visible-to-user",
    "draw": "draw",
}


def _text(v):
    """RICO stores content-desc as a list (often [None]); text as a string or None."""
    if isinstance(v, list):
        v = " ".join(x for x in v if isinstance(x, str))
    return v if isinstance(v, str) and v else None


def parse_screen(screen_id, root, labels):
    """Flatten one hierarchy. `labels` maps pointer -> CLAY label for this screen.

    Returns a list of dict rows. `clay_parent_idx` is the nearest ancestor that CLAY labelled
    (the parent in CLAY's slimmer tree); -1 when there is none.
    """
    rows, seen = [], set()
    # iterative pre-order DFS: (node, parent_idx, nearest_labelled_ancestor_idx, depth)
    stack = [(root, -1, -1, 0)]
    while stack:
        node, parent, lab_anc, depth = stack.pop()
        idx = len(rows)
        ptr = node.get("pointer")
        dup = ptr in seen
        seen.add(ptr)
        label = UNLABELLED if dup else labels.get(ptr, UNLABELLED)
        b = node.get("bounds") or [0, 0, 0, 0]
        x0, y0, x1, y1 = (int(v) for v in b)
        kids = [c for c in (node.get("children") or []) if c]
        row = {
            "screen_id": screen_id,
            "node_idx": idx,
            "parent_idx": parent,
            "clay_parent_idx": lab_anc,
            "depth": depth,
            "n_children": len(kids),
            "pointer": ptr,
            "dup_pointer": dup,
            "label": label,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "class_name": node.get("class"),
            "resource_id": node.get("resource-id"),
            "text": _text(node.get("text")),
            "content_desc": _text(node.get("content-desc")),
            "visibility": node.get("visibility"),
        }
        for col, key in FLAGS.items():
            row[col] = bool(node.get(key, False))
        rows.append(row)
        child_anc = idx if label != UNLABELLED else lab_anc
        for c in reversed(kids):  # reversed so children pop in document order
            stack.append((c, idx, child_anc, depth + 1))
    return rows


def add_derived(df):
    """Normalised box and a validity flag (positive area and overlapping the frame)."""
    df["nx0"] = (df.x0 / FRAME_W).astype("float32")
    df["ny0"] = (df.y0 / FRAME_H).astype("float32")
    df["nx1"] = (df.x1 / FRAME_W).astype("float32")
    df["ny1"] = (df.y1 / FRAME_H).astype("float32")
    df["box_ok"] = ((df.x1 > df.x0) & (df.y1 > df.y0) & (df.x1 > 0) & (df.y1 > 0)
                    & (df.x0 < FRAME_W) & (df.y0 < FRAME_H))
    df["is_labelled"] = df.label != UNLABELLED
    return df


DTYPES = {
    "screen_id": "int32", "node_idx": "int32", "parent_idx": "int32", "clay_parent_idx": "int32",
    "depth": "int16", "n_children": "int16", "label": "int8",
    "x0": "int32", "y0": "int32", "x1": "int32", "y1": "int32",
}


def load_screen(rico_dir, screen_id, labels):
    """Read one RICO json and return (node DataFrame, screen-level dict)."""
    with open(Path(rico_dir) / f"{screen_id}.json", encoding="utf8") as f:
        d = json.load(f)
    rows = parse_screen(screen_id, d["activity"]["root"], labels)
    df = add_derived(pd.DataFrame(rows).astype(DTYPES))
    meta = {
        "screen_id": screen_id,
        "activity_name": d.get("activity_name"),
        "keyboard": bool(d.get("is_keyboard_deployed", False)),
        "n_nodes": len(df),
        "n_labelled": int(df.is_labelled.sum()),
        "n_dup_pointer_rows": int(df.dup_pointer.sum()),
        "n_labels_unmatched": len(set(labels) - set(df.pointer[df.is_labelled])),
    }
    return df, meta
