"""Training and scoring for the type head (Model 1). Hyperparameters come from configs/train_element_types_metadata_mlp.yaml.

Model selection: early stopping on clean-dev macro_f1_types (the primary metric), patience in epochs;
the best epoch's weights are kept. Test is never touched here.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from uistruct.data.features import N_CLASSES
from uistruct.eval.metrics import f1_from_confusion
from uistruct.models.type_mlp import TypeMLP


# ---------------------------------------------------------------- scoring from confusion matrices
def per_screen_confusion(screen_ids, y, p):
    """(n_screens, 24, 24) confusion matrices, rows = truth, cols = prediction, labels 0..23."""
    _, s = np.unique(screen_ids, return_inverse=True)
    cm = np.zeros((s.max() + 1, N_CLASSES, N_CLASSES), np.int32)
    np.add.at(cm, (s, y, p), 1)
    return cm


def scores(cm):
    """cm: summed (24, 24) confusion. Same definitions as eval.metrics.type_metrics."""
    full = np.zeros((N_CLASSES + 1, N_CLASSES + 1), np.int64)
    full[:N_CLASSES, :N_CLASSES] = cm
    f1 = f1_from_confusion(full, N_CLASSES)
    return {"macro_f1_types": float(np.nanmean(f1[1:])), "invalid_f1": float(f1[0])}


def macro_types(cm):
    return scores(cm)["macro_f1_types"]


def invalid_f1(cm):
    return scores(cm)["invalid_f1"]


# ---------------------------------------------------------------- MLP
def _tensors(a):
    return [torch.from_numpy(a[k]) for k in ("num", "name", "pieces", "y")]


def predict(model, a, batch=65536):
    model.eval()
    num, name, pieces, _ = _tensors(a)
    out = []
    with torch.no_grad():
        for i in range(0, len(num), batch):
            out.append(model(num[i:i + batch], name[i:i + batch], pieces[i:i + batch]).softmax(1))
    return torch.cat(out).numpy()


def train_mlp(hp, enc, tr, dv, dev_screens, seed, log=print):
    """hp: dict of hyperparameters. tr/dv: arrays from Encoder.transform. Returns model, history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TypeMLP(tr["num"].shape[1], len(enc.names) + 2, len(enc.pieces) + 2,
                    use_name=hp["use_name"], use_pieces=hp["use_pieces"], emb_dim=hp["emb_dim"],
                    hidden=hp["hidden"], layers=hp["layers"], dropout=hp["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(enc.class_weights(hp["class_weight"]), dtype=torch.float32))
    num, name, pieces, y = _tensors(tr)
    g = torch.Generator().manual_seed(seed)
    best, best_state, hist, wait = -1.0, None, [], 0
    for ep in range(hp["max_epochs"]):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(y), generator=g)
        tot = 0.0
        for i in range(0, len(y), hp["batch"]):
            b = perm[i:i + hp["batch"]]
            opt.zero_grad()
            loss = loss_fn(model(num[b], name[b], pieces[b]), y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        p = predict(model, dv).argmax(1)
        sc = scores(per_screen_confusion(dev_screens, dv["y"], p).sum(0))
        hist.append({"epoch": ep + 1, "train_loss": tot / len(y), **sc, "sec": round(time.time() - t0, 1)})
        log(f"  ep {ep + 1:2d} loss {tot / len(y):.4f} dev macro {sc['macro_f1_types']:.4f} inv {sc['invalid_f1']:.4f} ({time.time() - t0:.1f}s)")
        if sc["macro_f1_types"] > best:
            best, best_state, wait = sc["macro_f1_types"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return model, hist


def sample_hp(space, rng):
    """One random configuration from a search space {name: list of choices | {loguniform: [lo, hi]}}."""
    hp = {}
    for k, v in space.items():
        if isinstance(v, dict) and "loguniform" in v:
            lo, hi = v["loguniform"]
            hp[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            hp[k] = v[rng.integers(len(v))]
    return hp
