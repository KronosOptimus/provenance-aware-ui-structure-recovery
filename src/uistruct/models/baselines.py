"""Non-neural type-head baselines.

majority   every element gets the most frequent training label (floor)
lookup     class-name lookup: most frequent training label for the class name's last part; unseen
           names get the training majority (as in the data analysis, here on the clean splits and the report metrics)
gbdt       gradient-boosted trees (sklearn HistGradientBoosting) on the same inputs as the MLP: numeric
           features, the class name as a categorical feature (most frequent 254 names, rest = "other"),
           and indicators for the 64 most frequent camel-case pieces
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from uistruct.data.features import camel_pieces

MAX_CATS = 254
N_PIECE_FLAGS = 64


def majority(train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), train.label.value_counts().idxmax())


def lookup(train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    lut = train.groupby("name").label.agg(lambda x: x.value_counts().idxmax())
    return df.name.map(lut).fillna(train.label.value_counts().idxmax()).astype(int).to_numpy()


class GBDTInputs:
    def __init__(self, numeric):
        self.numeric = list(numeric)

    def fit(self, train):
        self.cats = {n: i for i, n in enumerate(train.name.value_counts().index[:MAX_CATS])}
        pc = {}
        for n, c in train.name.value_counts().items():
            for p in camel_pieces(n):
                pc[p] = pc.get(p, 0) + int(c)
        self.piece_list = [p for p, _ in sorted(pc.items(), key=lambda t: -t[1])[:N_PIECE_FLAGS]]
        return self

    def transform(self, df):
        cat = df.name.map(self.cats).fillna(MAX_CATS).to_numpy(np.float64)[:, None]
        pidx = {p: j for j, p in enumerate(self.piece_list)}
        flags = np.zeros((len(df), len(self.piece_list)))
        for n, idx in df.groupby("name").indices.items():
            for p in camel_pieces(n):
                if p in pidx:
                    flags[idx, pidx[p]] = 1
        return np.hstack([df[self.numeric].to_numpy(np.float64), cat, flags])

    @property
    def categorical_mask(self):
        return [False] * len(self.numeric) + [True] + [False] * len(self.piece_list)


def gbdt(hp, seed):
    return HistGradientBoostingClassifier(
        learning_rate=hp["learning_rate"], max_leaf_nodes=hp["max_leaf_nodes"], l2_regularization=hp["l2"],
        min_samples_leaf=hp["min_samples_leaf"], class_weight=hp["class_weight"], max_iter=hp["max_iter"],
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=seed)
