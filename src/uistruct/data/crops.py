"""Per-element image crops for the CNN (Model 3).

As for the colour features, every screenshot is first resized to 540 x 960 (removes the half-resolution confound). Each
element's crop is its box plus a fixed margin of MARGIN px on every side (context: lets the CNN see a
box that is misaligned with what is drawn), clipped to the frame, then resized to a square of `size` px
(aspect ratio is not kept; the box shape is already a numeric input). Landscape screenshots and boxes
with no visible pixels give an all-zero crop (they carry the `no_image` flag of the colour features).

Crops are cut once and kept as one uint8 array (n_elements, size, size, 3) in the row order of
`features.load_elements` (screens contiguous), so training reads contiguous chunks and never decodes JPEGs
(Colab has 2 CPU cores).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SIZE = (540, 960)
MARGIN = 8
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_box(box, margin=MARGIN, size=SIZE):
    """box (nx0, ny0, nx1, ny1) in the 0-1 frame -> pixel box (x0, y0, x1, y1) with margin, clipped,
    or None when the box has no visible pixels."""
    W, H = size
    x0, y0, x1, y1 = (int(round(box[0] * W)), int(round(box[1] * H)), int(round(box[2] * W)), int(round(box[3] * H)))
    if min(x1, W) - max(x0, 0) < 1 or min(y1, H) - max(y0, 0) < 1:
        return None
    return max(x0 - margin, 0), max(y0 - margin, 0), min(x1 + margin, W), min(y1 + margin, H)


def screen_crops(path, boxes, sizes, margin=MARGIN):
    """All elements of one screen. Returns {size: (n, size, size, 3) uint8}."""
    from PIL import Image
    out = {s: np.zeros((len(boxes), s, s, 3), np.uint8) for s in sizes}
    im = Image.open(path).convert("RGB")
    if im.width > im.height:          # landscape: boxes are in the portrait frame and do not align
        return out
    im = im.resize(SIZE, Image.BILINEAR)
    for i, b in enumerate(boxes):
        pb = crop_box(b, margin)
        if pb is None:
            continue
        c = im.crop(pb)
        for s in sizes:
            out[s][i] = np.asarray(c.resize((s, s), Image.BILINEAR))
    return out


def _job(a):
    path, boxes, sizes, margin = a
    return screen_crops(path, boxes, sizes, margin)


def iter_screen_crops(df, img_dir, sizes, workers=2, margin=MARGIN, window=512):
    """Yields (row_start, row_end, {size: crops}) screen by screen, in df row order.
    df must have screens contiguous (load_elements order) and columns screen_id, nx0..ny1."""
    sid = df.screen_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1, len(sid)]
    boxes = df[["nx0", "ny0", "nx1", "ny1"]].to_numpy(np.float64)
    spans = list(zip(starts[:-1], starts[1:]))
    with ProcessPoolExecutor(workers) as ex:
        for w in range(0, len(spans), window):   # bounded windows: finished crops never pile up in RAM
            part = spans[w:w + window]
            jobs = [(f"{img_dir}/{sid[a]}.jpg", boxes[a:b], tuple(sizes), margin) for a, b in part]
            for (a, b), crops in zip(part, ex.map(_job, jobs, chunksize=16)):
                yield a, b, crops


def build_cache(df, img_dir, size, out_path, workers=2, margin=MARGIN, log=print):
    """Cut every element's crop into a uint8 .npy memmap of shape (len(df), size, size, 3)."""
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(len(df), size, size, 3))
    for k, (a, b, crops) in enumerate(iter_screen_crops(df, img_dir, [size], workers, margin)):
        arr[a:b] = crops[size]
        if k % 5000 == 0:
            log(f"  cache {size}px: row {b:,} / {len(df):,}")
    arr.flush()
    return out_path
