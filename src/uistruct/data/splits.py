"""Leakage-free splits.

CLAY's official split is by screen, so one company's apps can sit in train and in dev/test. Screens of
one company share templates, styles and custom class names: 35% of CLAY's test screens come from a
company with screens in CLAY's train set. A model could memorise those and look better than it is.

Rule (final; do not keep trimming):
  1. dev and test stay exactly as CLAY published them;
  2. a train screen is dropped if its company appears in dev or test, or if its labelled layout is
     identical to a dev/test screen's layout (second leakage route: copied screens);
  3. a dev screen is dropped if its company appears in test (model selection never sees a test company).

Company = first two parts of the package name (`com.andromo.dev123.app` -> `com.andromo`); package =
`activity_name` before '/'. The rule is crude on purpose: it over-groups country prefixes like `br.com`,
which only costs training data and can never add leakage. Generic layouts shared by unrelated apps
(e.g. toolbar + icon + title) are not leakage and are not removed beyond the identical-layout rule.
"""
import numpy as np
import pandas as pd


def app_of(activity_name: pd.Series) -> pd.Series:
    return activity_name.str.split("/").str[0]


def company_of(app: pd.Series) -> pd.Series:
    return app.str.split(".").str[:2].str.join(".")


def clean_splits(screens: pd.DataFrame, layout_hash: pd.Series = None,
                 drop_dev_test_overlap: bool = True) -> pd.DataFrame:
    """Return screens with `app`, `company`, `split` (CLAY official) and `split_clean`
    (train/dev/test/dropped). `layout_hash`: optional Series indexed by screen_id."""
    s = screens[["screen_id", "split", "activity_name"]].copy()
    s["app"] = app_of(s.activity_name)
    s["company"] = company_of(s.app)
    ev = s.split.isin(["dev", "test"])
    leak = s.company.isin(set(s[ev].company))
    if layout_hash is not None:
        h = s.screen_id.map(layout_hash)
        leak |= h.isin(set(h[ev].dropna()))
    s["split_clean"] = s.split
    s.loc[(s.split == "train") & leak, "split_clean"] = "dropped"
    if drop_dev_test_overlap:
        s.loc[(s.split == "dev") & s.company.isin(set(s[s.split == "test"].company)), "split_clean"] = "dropped"
    return s


def nested_company_subsets(train: pd.DataFrame, fracs, seed=0) -> dict:
    """Learning-curve subsets of the training screens, grown by whole companies.

    train: one row per training screen with `screen_id` and `company`. Companies are shuffled once
    (seed); subset f = the first companies whose screens reach f of all training screens. Subsets are
    nested, and no company is split, so every subset is as company-disjoint from dev/test as the full set.
    Returns {frac: set of screen ids}.
    """
    comp = train.groupby("company").screen_id.apply(list)
    order = np.random.default_rng(seed).permutation(len(comp))
    cum = np.cumsum([len(comp.iloc[i]) for i in order])
    out = {}
    for f in fracs:
        k = int(np.searchsorted(cum, f * cum[-1])) + 1
        out[f] = {s for i in order[:k] for s in comp.iloc[i]}
    return out


def overlap_report(s: pd.DataFrame, col: str, unit: str = "company") -> dict:
    """Shared apps/companies between each pair of splits in column `col`."""
    groups = {k: set(g[unit]) for k, g in s.groupby(col) if k != "dropped"}
    out = {}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        shared = groups.get(a, set()) & groups.get(b, set())
        out[f"{a}-{b}"] = {f"shared_{unit}": len(shared),
                           f"{a}_screens": int(((s[col] == a) & s[unit].isin(shared)).sum()),
                           f"{b}_screens": int(((s[col] == b) & s[unit].isin(shared)).sum())}
    return out
