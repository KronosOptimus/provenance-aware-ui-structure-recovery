"""Raw view-hierarchy helpers for the metadata nesting conditions.

Copied unchanged from scripts/structure_rule_baselines.py so package code can use them; the
script keeps its own copy because its runs record that code. tests/test_fusion_scorer.py checks both agree.
"""
import pandas as pd

from uistruct.eval.metrics import VROOT


def prune(parents: dict, leaves: set) -> dict:
    """Keep the given leaves and their ancestors; re-id non-leaf nodes as ("raw", idx)."""
    keep = set()
    for l in leaves:
        n = l
        while n != VROOT and n not in keep:
            keep.add(n)
            n = parents[n]
    name = lambda n: n if (n in leaves or n == VROOT) else ("raw", n)  # noqa: E731
    return {name(n): name(parents[n]) for n in keep}


def raw_chain(raw: pd.DataFrame, leaves: set) -> dict:
    """Raw hierarchy with unusable-box nodes skipped (their children move up)."""
    par = dict(zip(raw.node_idx, raw.parent_idx))
    ok = dict(zip(raw.node_idx, raw.box_ok))
    out = {}
    for n in raw.node_idx:
        if not ok[n] and n not in leaves:
            continue
        p = par[n]
        while p != VROOT and not ok[p]:
            p = par[p]
        out[n] = p
    return out
