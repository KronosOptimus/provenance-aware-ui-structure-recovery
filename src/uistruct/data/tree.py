"""Ground-truth elements and tree on CLAY.

Elements = CLAY-labelled nodes except ROOT. Valid elements have a type (label 1..23); invalid ones
(label 0, BACKGROUND) are the metadata errors and have no place in the true tree.

True parent of a valid element = its nearest valid ancestor in CLAY's tree (follow `clay_parent_idx`,
skipping invalid nodes, whose valid children move up). Elements whose chain ends at ROOT, or at no
labelled node, hang from a virtual root (parent -1). Using a virtual root instead of the ROOT node
treats the 23 screens without a ROOT label the same way as the others. Edges to the virtual root are
real predictions (is this element top-level?) and are scored.
"""
import numpy as np
import pandas as pd

VROOT = -1       # parent id of top-level elements
NO_PARENT = -2   # invalid elements: not in the true tree
ROOT_LABEL, INVALID_LABEL = -1, 0


def build_elements(nodes: pd.DataFrame) -> pd.DataFrame:
    """nodes: the node table (needs screen_id, node_idx, clay_parent_idx, label, is_labelled).

    Returns one row per element: screen_id, node_idx, label, is_valid, gt_parent (node_idx of the
    true parent, VROOT, or NO_PARENT for invalid elements). Extra columns of `nodes` are carried over.
    """
    lab = nodes[nodes.is_labelled].sort_values(["screen_id", "node_idx"]).reset_index(drop=True)
    key = lab.screen_id.to_numpy(np.int64) * 10_000 + lab.node_idx.to_numpy()
    pkey = lab.screen_id.to_numpy(np.int64) * 10_000 + lab.clay_parent_idx.to_numpy()
    pos = np.searchsorted(key, pkey)
    has_p = lab.clay_parent_idx.to_numpy() >= 0
    pos = np.where(has_p, np.minimum(pos, len(key) - 1), -1)
    if (has_p & (key[np.maximum(pos, 0)] != pkey)).any():
        raise ValueError("clay_parent_idx points to a node that is not labelled")
    label = lab.label.to_numpy()
    node_idx = lab.node_idx.to_numpy()
    # nearest valid ancestor; pre-order means a parent row always precedes its children
    anc = np.full(len(lab), VROOT, dtype=np.int64)
    for i in range(len(lab)):
        p = pos[i]
        if p < 0:
            continue
        anc[i] = node_idx[p] if label[p] > 0 else anc[p]
    lab["is_valid"] = label > 0
    lab["gt_parent"] = np.where(label > 0, anc, NO_PARENT)
    return lab[label != ROOT_LABEL].reset_index(drop=True)
