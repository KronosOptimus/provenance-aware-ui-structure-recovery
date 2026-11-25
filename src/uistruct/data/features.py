"""Per-element input features for the type head (Model 1).

Inputs come only from the raw view hierarchy (metadata) and the element's box:
  box    cx, cy, w, h, log area, log aspect (normalised 0-1 frame, clipped to [-0.5, 1.5])
  tree   raw depth and log(1 + raw child count). The RAW tree, never CLAY's tree: CLAY's tree is built
         from the labelled nodes, so its depth/child counts would leak CLAY's labelling.
  flags  the nine raw boolean attributes
  name   Android class name, last dotted part (e.g. "AppCompatImageButton"), one token
  pieces the same name split at camel-case ("app compat image button"), a bag of tokens; lets an unseen
         custom class (e.g. "MyFancyTextView") share "text" and "view" with known ones

  colour the 18 crop colour statistics of src/uistruct/data/colour.py (Model 2), cached in
         data/processed/colour_features.parquet by scripts/extract_colour_features.py

Everything fitted (scaler, vocabularies, class weights) is fitted on the training rows only.
"""
import re

import numpy as np
import pandas as pd

from uistruct.data.colour import COLOUR

BOX = ["cx", "cy", "w", "h", "log_area", "log_aspect"]
TREE = ["depth", "log_children"]
FLAGS = ["clickable", "long_clickable", "focusable", "scrollable_v", "scrollable_h", "enabled", "selected",
         "visible_to_user", "draw"]
FEATURE_SETS = {  # numeric columns, and whether the class name is used
    "box": (BOX, False),
    "box_tree_flags": (BOX + TREE + FLAGS, False),
    "model1": (BOX + TREE + FLAGS, True),
    "box_colour": (BOX + COLOUR, False),             # pixels + box, no metadata
    "model2": (BOX + TREE + FLAGS + COLOUR, True),   # Model 1 + colour
    # Model 3: the CNN crop embedding is a separate input (models/cnn.py); no_image marks an all-zero crop
    "model1_cnn": (BOX + TREE + FLAGS + ["no_image"], True),   # Model 3: Model 1 + CNN (replaces colour)
    "model2_cnn": (BOX + TREE + FLAGS + COLOUR, True),         # Model 2 + CNN (colour kept)
    "box_cnn": (BOX + ["no_image"], False),                    # pixels + box, no metadata
}
N_CLASSES = 24          # 0 = invalid, 1..23 = types (ROOT is not an element)
PAD, UNK = 0, 1
MAX_PIECES = 6
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")


def camel_pieces(name: str) -> list:
    return [p.lower() for p in _CAMEL.findall(name)][:MAX_PIECES]


def load_elements(data_dir="data/processed", splits=("train", "dev", "test"), colour=False) -> pd.DataFrame:
    """Elements of the clean splits joined to their raw node attributes, with derived features.
    colour=True adds the colour columns (joined after sorting, so row order equals colour=False)."""
    el = pd.read_parquet(f"{data_dir}/elements.parquet",
                         columns=["screen_id", "node_idx", "label", "is_valid", "split_clean"])
    el = el[el.split_clean.isin(splits)]
    nodes = pd.read_parquet(f"{data_dir}/nodes.parquet",
                            columns=["screen_id", "node_idx", "depth", "n_children", "class_name",
                                     "nx0", "ny0", "nx1", "ny1", *FLAGS])
    df = el.merge(nodes, on=["screen_id", "node_idx"], how="left", validate="one_to_one")
    b = df[["nx0", "ny0", "nx1", "ny1"]].to_numpy(np.float64).clip(-0.5, 1.5)
    w, h = np.maximum(b[:, 2] - b[:, 0], 1e-4), np.maximum(b[:, 3] - b[:, 1], 1e-4)
    df["cx"], df["cy"], df["w"], df["h"] = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, w, h
    df["log_area"], df["log_aspect"] = np.log(w * h), np.log(w / h)
    df["log_children"] = np.log1p(df.n_children.astype(float))
    df["depth"] = df.depth.astype(float)
    df["name"] = df.class_name.fillna("").str.split(".").str[-1]
    # position order (top, left) inside each screen; never the view-hierarchy order
    df = df.sort_values(["screen_id", "cy", "cx"]).reset_index(drop=True)
    if colour:
        col = pd.read_parquet(f"{data_dir}/colour_features.parquet")
        df = df.merge(col, on=["screen_id", "node_idx"], how="left", validate="one_to_one")
        assert not df[COLOUR].isna().any().any(), "colour features missing for some elements"
    return df


class Encoder:
    """Fit on training rows; transform any split into model arrays."""

    def __init__(self, numeric, min_count=5):
        self.numeric, self.min_count = list(numeric), min_count

    def fit(self, df):
        x = df[self.numeric].to_numpy(np.float64)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        vc = df.name.value_counts()
        self.names = {n: i + 2 for i, n in enumerate(vc[vc >= self.min_count].index)}
        pc = {}
        for n, c in vc.items():
            for p in camel_pieces(n):
                pc[p] = pc.get(p, 0) + int(c)
        pc = sorted(((c, p) for p, c in pc.items() if c >= self.min_count), reverse=True)
        self.pieces = {p: i + 2 for i, (_, p) in enumerate(pc)}
        counts = np.bincount(df.label.to_numpy(), minlength=N_CLASSES)
        self.class_counts = counts
        return self

    def class_weights(self, kind):
        c = np.maximum(self.class_counts, 1).astype(float)
        f = {"none": np.ones_like(c), "inv_sqrt": 1 / np.sqrt(c), "inv": 1 / c}[kind]
        return f / (f * c).sum() * c.sum()   # weighted mean over training rows = 1

    def transform(self, df):
        x = ((df[self.numeric].to_numpy(np.float64) - self.mean) / self.std).astype(np.float32)
        name = df.name.map(self.names).fillna(UNK).to_numpy(np.int64)
        uniq = {n: [self.pieces.get(p, UNK) for p in camel_pieces(n)] for n in df.name.unique()}
        pieces = np.zeros((len(df), MAX_PIECES), np.int64)
        for n, idx in df.groupby("name").indices.items():
            ids = uniq[n]
            if ids:
                pieces[idx, :len(ids)] = ids
        return {"num": x, "name": name, "pieces": pieces, "y": df.label.to_numpy(np.int64)}
