"""Type-head report for a set of models scored on the same rows (same definitions as scripts/train_element_types_with_colour.py).

For every model: macro-F1 over 23 types and invalid F1 (CI by resampling screens through per-screen confusion
matrices) and, for probability outputs, invalid AUROC (CI by resampling screens). For every pair: paired
bootstrap differences on the same resamples.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from uistruct.eval.metrics import TYPES, type_metrics
from uistruct.eval.stats import bootstrap_rows, bootstrap_stat, paired_bootstrap_stat
from uistruct.train.type_head import invalid_f1, macro_types, per_screen_confusion


def report(sid, y, probs, pairs, labels, per_class_pair=None, n_boot=1000, seed=0):
    """probs: {name: (n, 24) probabilities, or (n,) hard labels}. labels: {class id: name}."""
    pred = {k: (p.argmax(1) if p.ndim == 2 else p) for k, p in probs.items()}
    inv = (y == 0).astype(int)
    out, cms = {"scores": {}}, {}
    for k, p in pred.items():
        cms[k] = per_screen_confusion(sid, y, p)
        out["scores"][k] = {"macro_f1_types": bootstrap_stat(cms[k], macro_types, n_boot, seed),
                            "invalid_f1": bootstrap_stat(cms[k], invalid_f1, n_boot, seed)}
        if probs[k].ndim == 2:
            pk = probs[k][:, 0]
            out["scores"][k]["invalid_auroc"] = bootstrap_rows(sid, lambda i, pk=pk: roc_auc_score(inv[i], pk[i]),
                                                               n_boot, seed)
    pairs = [(a, b) for a, b in pairs if a in cms and b in cms]
    out["paired_macro"] = {f"{a}_vs_{b}": paired_bootstrap_stat(cms[a], cms[b], macro_types, n_boot, seed)
                           for a, b in pairs}
    out["paired_invalid_f1"] = {f"{a}_vs_{b}": paired_bootstrap_stat(cms[a], cms[b], invalid_f1, n_boot, seed)
                                for a, b in pairs}
    out["paired_invalid_auroc"] = {}
    for a, b in pairs:
        if probs[a].ndim == 2 and probs[b].ndim == 2:
            pa, pb = probs[a][:, 0], probs[b][:, 0]
            out["paired_invalid_auroc"][f"{a}_vs_{b}"] = bootstrap_rows(
                sid, lambda i: roc_auc_score(inv[i], pa[i]), n_boot, seed,
                other=lambda i: roc_auc_score(inv[i], pb[i]))
    if per_class_pair and all(k in pred for k in per_class_pair):
        names = ["INVALID(BACKGROUND)"] + [labels[c] for c in TYPES]

        def pc(p):
            m = type_metrics(y, p)
            return dict(zip(names, [round(m["invalid_f1"], 4)] + [m["per_class_f1"][c] for c in TYPES]))
        a, b = (pc(pred[k]) for k in per_class_pair)
        out["per_class_f1"] = {"models": list(per_class_pair), **{
            n: {"a": a[n], "b": b[n], "delta": None if a[n] is None or b[n] is None else round(a[n] - b[n], 4)}
            for n in names}}
    return out
