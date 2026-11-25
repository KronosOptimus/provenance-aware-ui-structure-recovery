"""Crop embeddings for the nesting scorers of Models 4 and 5.

Each token gets the embedding of one image crop, cut exactly as for Model 3 (`data/crops.py`: screenshot
resized to 540 x 960, box + 8 px margin, clipped, squashed to 96 x 96 px):
  leaf                         its own box
  generated candidate (P, M)   the box around its member leaves
  raw candidate (M only)       the raw box the metadata claims
A screen's boxes are de-duplicated (a singleton's box is its leaf's box), cut once and embedded by every
backbone in one pass. Backbones (both ResNet-18, 512-d pooled output, ImageNet normalisation):
  model3    Model 3's fine-tuned backbone, kept frozen
  imagenet  the ImageNet weights only (never saw UI data)
Storage per split: `emb_<split>_<backbone>.npy` float16 (n_boxes, 512) and `emb_<split>_index.pkl`
(per screen: row offset + index arrays into the screen's boxes for leaves, pool P and pool M).
"""
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from uistruct.data.crops import screen_crops

SIZE = 96


def screen_boxes(sc, groups=None):
    """Unique crop boxes of one screen and, per token group, the index of each token's box.
    groups: {name: (n, 4) boxes}; default = leaves and the candidate pools. The provenance network passes every level's groups."""
    if groups is None:
        groups = {"leaf": sc["leaf_xyxy"], **{k: sc[k]["crop_xyxy"] for k in ("P", "M") if k in sc}}
    allb = np.vstack([np.asarray(b, np.float64).reshape(-1, 4) for b in groups.values()])
    uniq, inv = np.unique(np.round(allb, 6), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    idx, a = {}, 0
    for k, b in groups.items():
        n = len(np.asarray(b).reshape(-1, 4))
        idx[k] = inv[a:a + n].astype(np.int32)
        a += n
    return uniq, idx


def _crop_job(a):
    path, boxes = a
    return screen_crops(path, boxes, [SIZE])[SIZE]


@torch.no_grad()
def embed_screens(screens, img_dir, backbones, device, workers, batch=4096, log=print, groups_fn=None):
    """screens: cached screens of one split. backbones: {name: frozen trunk}. Returns
    ({name: float16 (n_boxes, 512)}, index list). groups_fn(screen) -> box groups (provenance network: every level)."""
    from uistruct.models.cnn import Normalise
    norm = Normalise().to(device)
    for m in backbones.values():
        m.to(device).eval()
    boxes, index, off = [], [], 0
    for sc in screens:
        u, idx = screen_boxes(sc, groups_fn(sc) if groups_fn else None)
        boxes.append(u)
        index.append({"screen_id": sc["screen_id"], "offset": off, **idx})
        off += len(u)
    out = {k: np.zeros((off, 512), np.float16) for k in backbones}
    t0, buf, row, done = time.time(), [], 0, 0

    def flush():
        nonlocal row, buf
        x = torch.from_numpy(np.concatenate(buf)).to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            xn = norm(x)
            for k, m in backbones.items():
                out[k][row:row + len(x)] = m(xn).half().cpu().numpy()
        row += len(x)
        buf = []

    jobs = [(f"{img_dir}/{sc['screen_id']}.jpg", b) for sc, b in zip(screens, boxes)]
    with ProcessPoolExecutor(workers) as ex:
        for w in range(0, len(jobs), 1024):                    # bounded windows: crops never pile up in RAM
            for crops in ex.map(_crop_job, jobs[w:w + 1024], chunksize=16):
                buf.append(crops)
                done += len(crops)
                if sum(len(c) for c in buf) >= batch:
                    flush()
            log(f"  {min(w + 1024, len(jobs))}/{len(jobs)} screens, {done:,} crops ({time.time() - t0:.0f}s)")
    if buf:
        flush()
    assert row == off
    return out, index


def save(out, index, emb_dir, split):
    for k, e in out.items():
        np.save(f"{emb_dir}/emb_{split}_{k}.npy", e)
    with open(f"{emb_dir}/emb_{split}_index.pkl", "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(emb_dir, split, backbone):
    """-> (float16 array (n_boxes, 512), {screen_id: index dict})."""
    e = np.load(f"{emb_dir}/emb_{split}_{backbone}.npy", mmap_mode="r")
    index = pickle.load(open(f"{emb_dir}/emb_{split}_index.pkl", "rb"))
    return e, {d["screen_id"]: d for d in index}


def token_crops(e, idx, pool):
    """Crop embeddings (float16) for the leaves and the pool's candidates of one screen."""
    o = idx["offset"]
    return np.asarray(e[o + idx["leaf"]]), np.asarray(e[o + idx[pool]])
