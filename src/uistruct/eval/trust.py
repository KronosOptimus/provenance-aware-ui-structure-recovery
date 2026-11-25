"""Trust scores of the provenance network: ranking, accuracy of the probabilities, calibration.

p = predicted probability that a metadata box is faithful; y = 1 if it is (CLAY truth).
Ranking uses "unfaithful = positive" (score 1 - p), the error the trust head has to find.
ECE: 15 equal-width bins and 15 equal-count bins. Brier skill = 1 - Brier / Brier of a constant base rate.
Calibration maps are fitted on logits: temperature (one scale) and Platt (scale + shift); both by
maximum likelihood, on data given by the caller (validation only; never test).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _nll(z, y):
    return float(np.mean(np.logaddexp(0, z) - y * z))


def fit_temperature(p, y):
    """T > 0 minimising NLL of sigmoid(logit(p) / T)."""
    z, y = logit(p), np.asarray(y, float)
    r = minimize(lambda t: _nll(z / np.exp(t[0]), y), [0.0], method="L-BFGS-B", bounds=[(-4, 4)])
    return {"kind": "temperature", "T": float(np.exp(r.x[0]))}


def fit_platt(p, y):
    """a, b minimising NLL of sigmoid(a * logit(p) + b)."""
    z, y = logit(p), np.asarray(y, float)
    r = minimize(lambda w: _nll(w[0] * z + w[1], y), [1.0, 0.0], method="L-BFGS-B")
    return {"kind": "platt", "a": float(r.x[0]), "b": float(r.x[1])}


def apply_cal(cal, p):
    z = logit(p)
    if cal is None:
        return np.clip(np.asarray(p, float), 0, 1)
    if cal["kind"] == "temperature":
        z = z / cal["T"]
    else:
        z = cal["a"] * z + cal["b"]
    return 1 / (1 + np.exp(-z))


def ece(p, y, n_bins=15, equal_count=False):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if equal_count:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    else:
        b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = 0.0
    for k in np.unique(b):
        m = b == k
        out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def reliability(p, y, n_bins=15):
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return [{"lo": k / n_bins, "hi": (k + 1) / n_bins, "n": int((b == k).sum()),
             "conf": float(p[b == k].mean()), "acc": float(y[b == k].mean())} for k in range(n_bins) if (b == k).any()]


def trust_scores(p, y, base_rate, n_bins=15):
    """All §5 numbers for one set of probabilities. base_rate = validation share of faithful boxes."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    both = 0 < y.mean() < 1
    brier = float(((p - y) ** 2).mean())
    brier0 = float(((base_rate - y) ** 2).mean())
    return {"n": int(len(y)), "share_unfaithful": float(1 - y.mean()),
            "auroc": float(roc_auc_score(1 - y, 1 - p)) if both else float("nan"),
            "auprc_unfaithful": float(average_precision_score(1 - y, 1 - p)) if both else float("nan"),
            "brier": brier, "brier_skill": float(1 - brier / brier0) if brier0 > 0 else float("nan"),
            "ece": ece(p, y, n_bins), "ece_equal_count": ece(p, y, n_bins, equal_count=True)}


def auroc_rows(y, p):
    """stat(idx) for eval.stats.bootstrap_rows: AUROC (unfaithful positive) on the rows idx."""
    y, p = np.asarray(y, float), np.asarray(p, float)

    def f(i):
        yy = y[i]
        return roc_auc_score(1 - yy, 1 - p[i]) if 0 < yy.mean() < 1 else float("nan")
    return f


def split_halves(screen_ids, seed):
    """Two halves of the screens (validation calibration: fit on one half, score on the other)."""
    u = np.unique(screen_ids)
    a = set(np.random.default_rng(seed).permutation(u)[:len(u) // 2].tolist())
    return np.array([s in a for s in screen_ids])


def cross_fitted(p, y, screen_ids, fit, seed):
    """Validation numbers for a calibration map: fit on one half of the screens, apply to the other, both ways."""
    h = split_halves(screen_ids, seed)
    out = np.empty(len(p))
    for side in (True, False):
        out[h == side] = apply_cal(fit(p[h != side], y[h != side]), p[h == side])
    return out
