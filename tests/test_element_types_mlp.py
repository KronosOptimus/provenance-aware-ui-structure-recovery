"""Model 1 tests: features, company subsets, confusion-based scores, paired bootstrap of a statistic."""
import numpy as np
import pandas as pd
import pytest

from uistruct.data.features import Encoder, camel_pieces
from uistruct.data.splits import nested_company_subsets
from uistruct.eval.metrics import type_metrics
from uistruct.eval.stats import paired_bootstrap_stat
from uistruct.train.type_head import macro_types, per_screen_confusion, scores


def test_camel_pieces():
    assert camel_pieces("AppCompatImageButton") == ["app", "compat", "image", "button"]
    assert camel_pieces("HTMLTextView2") == ["html", "text", "view", "2"]
    assert camel_pieces("") == []


def test_nested_company_subsets():
    tr = pd.DataFrame({"screen_id": range(10), "company": list("aabbbcdddd")})
    s = nested_company_subsets(tr, [0.25, 0.5, 0.75, 1.0], seed=0)
    assert s[0.25] <= s[0.5] <= s[0.75] <= s[1.0] == set(range(10))
    comp = dict(zip(tr.screen_id, tr.company))
    for f, ids in s.items():   # whole companies only
        cs = {comp[i] for i in ids}
        assert ids == set(tr[tr.company.isin(cs)].screen_id)
        assert len(ids) >= f * 10


def test_scores_match_type_metrics():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 24, 3000), rng.integers(0, 24, 3000)
    p[:1500] = y[:1500]
    sid = rng.integers(0, 50, 3000)
    s = scores(per_screen_confusion(sid, y, p).sum(0))
    m = type_metrics(y, p)
    assert s["macro_f1_types"] == pytest.approx(m["macro_f1_types"])
    assert s["invalid_f1"] == pytest.approx(m["invalid_f1"])


def test_paired_bootstrap_stat_sign():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 24, 4000)
    sid = rng.integers(0, 80, 4000)
    good, bad = y.copy(), rng.integers(0, 24, 4000)
    r = paired_bootstrap_stat(per_screen_confusion(sid, y, good), per_screen_confusion(sid, y, bad), macro_types, n_boot=200)
    assert r["diff"] > 0.8 and r["p_a_not_better"] == 0


def test_encoder_fits_on_train_only():
    tr = pd.DataFrame({"a": [0.0, 2.0], "name": ["TextView"] * 2, "label": [1, 1]})
    te = pd.DataFrame({"a": [100.0], "name": ["NeverSeenWidget"], "label": [0]})
    enc = Encoder(["a"], min_count=1).fit(tr)
    out = enc.transform(te)
    assert out["num"][0, 0] == pytest.approx(99.0, rel=1e-4)     # (100 - 1) / 1
    assert out["name"][0] == 1                                    # UNK
