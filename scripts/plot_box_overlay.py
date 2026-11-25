"""Visual check: draw CLAY-labelled boxes (label > 0) from nodes.parquet on screenshots.

Run:  python scripts/plot_box_overlay.py   -> results/figures/box_alignment.png
Shows one 1080x1920 and one 540x960 train screen (the 4th with 15-30 labelled nodes, deterministic).
"""
import pandas as pd
from PIL import Image, ImageDraw

s = pd.read_parquet("data/processed/screens.parquet")
n = pd.read_parquet("data/processed/nodes.parquet",
                    columns=["screen_id", "label", "is_labelled", "nx0", "ny0", "nx1", "ny1"])
ims = []
for w in (1080, 540):
    sid = int(s[(s.img_w == w) & (s.split == "train") & s.n_labelled.between(15, 30)].screen_id.iloc[3])
    im = Image.open(f"data/raw/rico/combined/{sid}.jpg").convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for r in n[(n.screen_id == sid) & n.is_labelled & (n.label > 0)].itertuples():
        d.rectangle([r.nx0 * W, r.ny0 * H, r.nx1 * W, r.ny1 * H], outline=(255, 0, 0), width=max(1, W // 300))
    ims.append(im.resize((405, 720)))
    print("screen", sid, f"{W}x{H}")
out = Image.new("RGB", (820, 720), "white")
out.paste(ims[0], (0, 0))
out.paste(ims[1], (415, 0))
out.save("results/figures/box_alignment.png")
