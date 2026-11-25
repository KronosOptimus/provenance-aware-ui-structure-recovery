"""Per-element colour features from the screenshot crop (Model 2).

Every screenshot is resized to one size (540 x 960) first: 22% of RICO screenshots are half resolution,
and resolution changes edge and JPEG-noise statistics (a confound, not a property of the element).
Boxes (nx0..ny1, 0-1 frame, unclipped) are clipped to the frame before cropping.

Features (18, all from pixels only; no labels, no metadata strings):
  mean_r mean_g mean_b, std_r std_g std_b   colour mean and spread per channel (0-255)
  lum_mean lum_std                          brightness mean and spread
  sat_mean                                  mean (max - min channel), a cheap saturation
  dom_frac                                  share of pixels in the most common colour bin (8 levels/channel)
  col_entropy                               entropy of the 512-bin colour histogram
  edge_x edge_y                             mean |brightness difference| between neighbouring pixels
  ring_contrast                             |crop mean colour - mean colour of a ring just outside the box|
  border_mean border_max                    contrast between strips just inside and just outside each box
                                            edge (as in the nesting-input measurements), mean and max over the edges not at the screen border
  border_missing                            1 if no edge could be measured (box too small or at the border)
  no_image                                  1 if the box has no visible pixels or the screenshot is landscape
                                            (35 screens whose boxes do not align); all other features 0
"""
import numpy as np

COLOUR = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "lum_mean", "lum_std", "sat_mean", "dom_frac",
          "col_entropy", "edge_x", "edge_y", "ring_contrast", "border_mean", "border_max", "border_missing", "no_image"]
SIZE = (540, 960)


def _border(img, x0, y0, x1, y1, w):
    """Mean |inside strip - outside strip| per measurable edge."""
    H, W = img.shape[:2]
    if x1 - x0 < 2 * w + 2 or y1 - y0 < 2 * w + 2:
        return []
    ya, yb, xa, xb = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
    out = []
    for ok, a, o in [
        (x0 - w >= 0, (slice(ya, yb), slice(x0, x0 + w)), (slice(ya, yb), slice(x0 - w, x0))),
        (x1 + w <= W, (slice(ya, yb), slice(x1 - w, x1)), (slice(ya, yb), slice(x1, x1 + w))),
        (y0 - w >= 0, (slice(y0, y0 + w), slice(xa, xb)), (slice(y0 - w, y0), slice(xa, xb))),
        (y1 + w <= H, (slice(y1 - w, y1), slice(xa, xb)), (slice(y1, y1 + w), slice(xa, xb))),
    ]:
        if not ok or yb <= ya or xb <= xa:
            continue
        ia, oa = img[a], img[o]
        if ia.shape == oa.shape and ia.size:
            out.append(float(np.abs(ia - oa).mean()))
    return out


def crop_features(img, box, strip=2, ring=4):
    """img: (H, W, 3) float32 0-255; box: (nx0, ny0, nx1, ny1). Returns a list in COLOUR order."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(round(box[0] * W)), int(round(box[1] * H)), int(round(box[2] * W)), int(round(box[3] * H)))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if xb - xa < 1 or yb - ya < 1:
        return [0.0] * (len(COLOUR) - 1) + [1.0]
    c = img[ya:yb, xa:xb]
    flat = c.reshape(-1, 3)
    mean, std = flat.mean(0), flat.std(0)
    lum = c.mean(2)
    q = (c // 32).astype(np.int32)
    p = np.bincount((q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]).ravel(), minlength=512) / flat.shape[0]
    nz = p[p > 0]
    ex = float(np.abs(np.diff(lum, axis=1)).mean()) if lum.shape[1] > 1 else 0.0
    ey = float(np.abs(np.diff(lum, axis=0)).mean()) if lum.shape[0] > 1 else 0.0
    rr = img[max(ya - ring, 0):min(yb + ring, H), max(xa - ring, 0):min(xb + ring, W)].reshape(-1, 3)
    n_in = flat.shape[0]
    ring_mean = (rr.sum(0) - flat.sum(0)) / (rr.shape[0] - n_in) if rr.shape[0] > n_in else mean
    b = _border(img, x0, y0, x1, y1, strip)
    return [*mean, *std, float(lum.mean()), float(lum.std()), float((c.max(2) - c.min(2)).mean()), float(p.max()),
            float(-(nz * np.log(nz)).sum()), ex, ey, float(np.abs(mean - ring_mean).mean()),
            float(np.mean(b)) if b else 0.0, float(np.max(b)) if b else 0.0, 0.0 if b else 1.0, 0.0]


def screen_features(path, boxes, strip=2, ring=4):
    """All elements of one screen. boxes: (n, 4) array. Returns (n, len(COLOUR)) float32."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > im.height:   # landscape screenshot: boxes are in the portrait frame and do not align
        out = np.zeros((len(boxes), len(COLOUR)), np.float32)
        out[:, -1] = 1
        return out
    img = np.asarray(im.resize(SIZE, Image.BILINEAR), dtype=np.float32)
    return np.array([crop_features(img, b, strip, ring) for b in boxes], np.float32).reshape(-1, len(COLOUR))
