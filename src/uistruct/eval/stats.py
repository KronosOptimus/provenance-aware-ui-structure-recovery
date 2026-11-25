"""Uncertainty for test results: bootstrap over screens.

Screens, not nodes, are resampled, because nodes of one screen are not independent.
`paired_bootstrap` uses the same resamples for two systems, so the CI of their difference accounts
for screens that are hard for both.
"""
import numpy as np


def bootstrap_mean(values, n_boot=1000, seed=0, alpha=0.05):
    """Mean of per-screen values (NaN dropped) with a percentile CI."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:          # metric undefined on every screen (e.g. precision when nothing is predicted)
        return {"mean": float("nan"), "std": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(1)
    return {"mean": float(v.mean()), "std": float(v.std()), "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "n": int(len(v))}


def bootstrap_stat(per_screen, stat, n_boot=1000, seed=0, alpha=0.05):
    """CI for a statistic that is not a mean of screens (e.g. macro-F1).

    per_screen: array (n_screens, ...) of additive per-screen counts (e.g. confusion matrices);
    stat: function of the summed counts.
    """
    x = np.asarray(per_screen)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.array([stat(x[rng.integers(0, n, n)].sum(0)) for _ in range(n_boot)])
    return {"value": float(stat(x.sum(0))), "lo": float(np.nanquantile(boots, alpha / 2)),
            "hi": float(np.nanquantile(boots, 1 - alpha / 2)), "n": int(n)}


def paired_bootstrap_stat(per_a, per_b, stat, n_boot=1000, seed=0, alpha=0.05):
    """Like paired_bootstrap, for a statistic of summed per-screen counts (e.g. macro-F1 from confusion
    matrices). per_a / per_b: same screens in the same order. Returns stat(A) - stat(B) with CI and the
    one-sided p-value that A is not better than B."""
    a, b = np.asarray(per_a), np.asarray(per_b)
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots.append(stat(a[i].sum(0)) - stat(b[i].sum(0)))
    boots = np.array(boots)
    return {"diff": float(stat(a.sum(0)) - stat(b.sum(0))), "lo": float(np.nanquantile(boots, alpha / 2)),
            "hi": float(np.nanquantile(boots, 1 - alpha / 2)), "p_a_not_better": float((boots <= 0).mean()),
            "n": int(n)}


def paired_bootstrap(a, b, n_boot=1000, seed=0, alpha=0.05):
    """Per-screen values of system A and B on the same screens. Returns mean(A - B) with CI and the
    one-sided bootstrap p-value that A is not better than B."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "p_a_not_better": float((boots <= 0).mean()),
            "n": int(len(d))}


def bootstrap_rows(groups, stat, n_boot=1000, seed=0, alpha=0.05, other=None):
    """CI for a statistic of row-level arrays (e.g. AUROC), resampling whole screens.

    groups: screen id per row; stat(idx) -> value for the rows idx (repeats allowed).
    other: optional second statistic on the same resamples; then returns stat - other (paired), with the
    one-sided p-value that the difference is <= 0.
    """
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv))]
    rows = [order[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    rng = np.random.default_rng(seed)
    n = len(rows)
    f = (lambda i: stat(i) - other(i)) if other is not None else stat
    boots = np.array([f(np.concatenate([rows[j] for j in rng.integers(0, n, n)])) for _ in range(n_boot)])
    out = {"value": float(f(np.arange(len(inv)))), "lo": float(np.nanquantile(boots, alpha / 2)),
           "hi": float(np.nanquantile(boots, 1 - alpha / 2)), "n": int(n)}
    if other is not None:
        out["p_a_not_better"] = float((boots <= 0).mean())
    return out
