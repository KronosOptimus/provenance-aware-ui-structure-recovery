"""Colour feature tests: colour features on synthetic images with hand-known answers."""
import numpy as np
import pytest

from uistruct.data.colour import COLOUR, crop_features

F = {n: i for i, n in enumerate(COLOUR)}


def img_with_box(bg=0.0, fg=200.0):
    img = np.full((100, 60, 3), bg, np.float32)
    img[20:40, 10:30] = fg            # box x 10..30, y 20..40
    return img


def box(x0, y0, x1, y1, W=60, H=100):
    return (x0 / W, y0 / H, x1 / W, y1 / H)


def test_uniform_crop():
    f = crop_features(img_with_box(), box(10, 20, 30, 40))
    assert f[F["mean_r"]] == pytest.approx(200) and f[F["std_g"]] == 0
    assert f[F["dom_frac"]] == 1 and f[F["col_entropy"]] == 0 and f[F["edge_x"]] == 0
    assert f[F["ring_contrast"]] == pytest.approx(200)        # ring is all background
    assert f[F["border_mean"]] == pytest.approx(200) and f[F["border_missing"]] == 0 and f[F["no_image"]] == 0


def test_box_on_flat_background_has_no_border():
    f = crop_features(img_with_box(), box(35, 50, 55, 80))     # region of plain background
    assert f[F["border_max"]] == 0 and f[F["ring_contrast"]] == 0


def test_half_and_half_crop():
    img = np.zeros((100, 60, 3), np.float32)
    img[:, 30:] = 255
    f = crop_features(img, box(20, 20, 40, 40))
    assert f[F["dom_frac"]] == pytest.approx(0.5)
    assert f[F["col_entropy"]] == pytest.approx(np.log(2))
    assert f[F["edge_y"]] == 0 and f[F["edge_x"]] > 0


def test_offscreen_and_tiny_boxes():
    f = crop_features(img_with_box(), (1.2, 1.2, 1.5, 1.5))
    assert f[F["no_image"]] == 1 and sum(f[:-1]) == 0
    g = crop_features(img_with_box(), box(10, 20, 13, 23))     # too small for border strips
    assert g[F["border_missing"]] == 1 and g[F["no_image"]] == 0


def test_box_partly_outside_is_clipped():
    f = crop_features(img_with_box(), (-0.5, 20 / 100, 30 / 60, 40 / 100))
    assert f[F["no_image"]] == 0 and 0 < f[F["mean_r"]] < 200


def test_bootstrap_rows_auroc():
    from sklearn.metrics import roc_auc_score
    from uistruct.eval.stats import bootstrap_rows
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(200), 5)
    y = rng.integers(0, 2, 1000)
    good, bad = y + rng.normal(0, 0.5, 1000), rng.normal(0, 1, 1000)
    r = bootstrap_rows(g, lambda i: roc_auc_score(y[i], good[i]), n_boot=200)
    assert r["value"] == pytest.approx(roc_auc_score(y, good)) and r["lo"] < r["value"] < r["hi"]
    d = bootstrap_rows(g, lambda i: roc_auc_score(y[i], good[i]), n_boot=200, other=lambda i: roc_auc_score(y[i], bad[i]))
    assert d["value"] > 0.2 and d["p_a_not_better"] == 0
