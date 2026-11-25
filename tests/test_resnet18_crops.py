"""Model 3 tests: crop geometry, crop cache, batching and one tiny train/predict pass of the CNN head (CPU)."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from uistruct.data.crops import MARGIN, build_cache, crop_box, screen_crops
from uistruct.data.features import FEATURE_SETS, Encoder
from uistruct.models.cnn import FEAT_DIM, CropTypeNet, Normalise
from uistruct.train import cnn_head as C


def test_crop_box_margin_and_clip():
    # box 100..200 x 300..400 px in the 540 x 960 frame
    b = (100 / 540, 300 / 960, 200 / 540, 400 / 960)
    assert crop_box(b) == (100 - MARGIN, 300 - MARGIN, 200 + MARGIN, 400 + MARGIN)
    assert crop_box((0.0, 0.0, 0.1, 0.1))[:2] == (0, 0)                 # clipped at the frame
    assert crop_box((1.2, 0.2, 1.5, 0.3)) is None                       # off screen
    assert crop_box((0.5, 0.5, 0.5, 0.6)) is None                       # zero width


@pytest.fixture
def screens(tmp_path):
    """Two portrait screenshots (one half resolution) and one landscape screenshot."""
    a = np.zeros((960, 540, 3), np.uint8)
    a[300:400, 100:200] = 255                                          # white square
    Image.fromarray(a).save(tmp_path / "1.jpg", quality=95)
    Image.fromarray(a).resize((270, 480)).save(tmp_path / "2.jpg", quality=95)
    Image.fromarray(np.full((540, 960, 3), 128, np.uint8)).save(tmp_path / "3.jpg")
    return tmp_path


def test_screen_crops_content(screens):
    b = np.array([[100 / 540, 300 / 960, 200 / 540, 400 / 960], [1.2, 0.2, 1.5, 0.3]])
    c = screen_crops(screens / "1.jpg", b, [32, 64])
    assert c[32].shape == (2, 32, 32, 3) and c[64].dtype == np.uint8
    centre = c[64][0, 16:48, 16:48].mean()
    assert centre > 240 and c[64][0, :2].mean() < 15                    # square inside, margin outside
    assert c[64][1].max() == 0                                          # off-screen box -> zero crop
    half = screen_crops(screens / "2.jpg", b, [64])[64]
    assert abs(half[0, 16:48, 16:48].mean() - centre) < 10               # resolution confound removed
    assert screen_crops(screens / "3.jpg", b, [32])[32].max() == 0       # landscape -> zero crops


def test_build_cache_rows_follow_df(screens, tmp_path):
    df = pd.DataFrame({"screen_id": [1, 1, 3], "nx0": [100 / 540, 0.5, 0.1], "ny0": [300 / 960, 0.5, 0.1],
                       "nx1": [200 / 540, 0.6, 0.2], "ny1": [400 / 960, 0.6, 0.2]})
    p = build_cache(df, screens, 16, tmp_path / "c.npy", workers=1, log=lambda s: None)
    arr = np.load(p, mmap_mode="r")
    assert arr.shape == (3, 16, 16, 3)
    assert arr[0].mean() > arr[1].mean() and arr[2].max() == 0


def test_chunk_batches_cover_each_row_once():
    rng = np.random.default_rng(0)
    for chunk in (1, 7):
        b = C.chunk_batches(100, 20, chunk, rng)
        assert sorted(np.concatenate(b).tolist()) == list(range(100))


def test_crop_source_reads_runs():
    arr = np.arange(20 * 2 * 2 * 3, dtype=np.uint8).reshape(20, 2, 2, 3)
    s = C.Source("crop", arr, torch.device("cpu"))
    idx = np.r_[5:9, 0:3, 15:20]
    assert np.array_equal(s.get(idx).numpy(), arr[idx])


def test_normalise_imagenet():
    x = torch.full((1, 4, 4, 3), 255, dtype=torch.uint8)
    y = Normalise()(x)
    assert y.shape == (1, 3, 4, 4)
    assert y[0, 0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229, rel=1e-4)


def _toy(n=300, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({k: r.normal(size=n) for k in FEATURE_SETS["model1_cnn"][0]})
    df["name"] = r.choice(["TextView", "ImageView", "Button"], n)
    df["label"] = r.integers(0, 24, n)
    return df


def test_train_and_predict_frozen_and_finetuned():
    df = _toy()
    enc = Encoder(FEATURE_SETS["model1_cnn"][0], min_count=1).fit(df)
    a = enc.transform(df)
    hp = {"use_name": True, "use_pieces": True, "emb_dim": 8, "hidden": 16, "layers": 2, "dropout": 0.0,
          "lr": 1e-3, "weight_decay": 0.0, "class_weight": "none", "batch": 64, "max_epochs": 2, "patience": 2,
          "amp": False, "lr_backbone": 1e-4}
    dev = torch.device("cpu")
    sid = np.repeat(np.arange(30), 10)
    feats = np.random.default_rng(1).normal(size=(len(df), FEAT_DIM)).astype(np.float16)
    m, h = C.train({**hp, "chunk": 1}, enc, a, a, C.Source("feat", feats, dev), C.Source("feat", feats, dev), sid,
                   0, dev, fine_tune=False, log=lambda s: None)
    p = C.predict(m, a, C.Source("feat", feats, dev), dev)
    assert p.shape == (len(df), 24) and np.allclose(p.sum(1), 1, atol=1e-4) and len(h) == 2
    crops = np.random.default_rng(2).integers(0, 255, (len(df), 32, 32, 3), dtype=np.uint8)
    m2 = CropTypeNet(a["num"].shape[1], len(enc.names) + 2, len(enc.pieces) + 2,
                     backbone=__import__("uistruct.models.cnn", fromlist=["x"]).resnet18_trunk(pretrained=False),
                     use_pieces=True, emb_dim=8, hidden=16, layers=2)
    out = m2(torch.from_numpy(a["num"][:4]), torch.from_numpy(a["name"][:4]), torch.from_numpy(a["pieces"][:4]),
             torch.from_numpy(crops[:4]))
    assert out.shape == (4, 24)
