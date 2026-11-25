"""Training for the Model 3 type head: MLP + CNN crop embedding, frozen or fine-tuned.

Same selection rule as Models 1 and 2: early stopping on clean-dev macro_f1_types, best epoch's weights kept;
test is never touched here.

Image input per split is either
  stored frozen features  float16 (n, 512), moved to the GPU whole (frozen use), or
  a crop cache            uint8 memmap (n, S, S, 3) of that split, rows in load_elements order.
Batches are built from shuffled chunks of `chunk` consecutive rows, so the crop cache is read in contiguous
slices (random single-row reads from disk would be the bottleneck). A background thread prepares the next
batches while the GPU trains.
"""
import queue
import threading
import time

import numpy as np
import torch
from torch import nn

from uistruct.models.cnn import CropTypeNet, resnet18_trunk
from uistruct.train.type_head import per_screen_confusion, scores


def chunk_batches(n, batch, chunk, rng):
    """List of index arrays: chunks of `chunk` consecutive rows, chunk order shuffled, ~batch rows each."""
    if chunk == 1:
        return np.array_split(rng.permutation(n), max(-(-n // batch), 1))
    starts = np.arange(0, n, chunk)
    rng.shuffle(starts)
    per = max(batch // chunk, 1)
    out = []
    for i in range(0, len(starts), per):
        out.append(np.concatenate([np.arange(s, min(s + chunk, n)) for s in starts[i:i + per]]))
    return out


class Source:
    """Image input of one split: 'feat' (array or tensor of stored features) or 'crop' (uint8 memmap)."""

    def __init__(self, kind, data, device):
        self.kind, self.device = kind, device
        self.data = torch.as_tensor(np.asarray(data)).to(device) if kind == "feat" else data

    def __len__(self):
        return len(self.data)

    def get(self, idx):
        if self.kind == "feat":
            return self.data[torch.as_tensor(idx, device=self.device)]
        # idx is a concatenation of consecutive runs: read each run as one slice
        cuts = np.flatnonzero(np.diff(idx) != 1) + 1
        parts = [self.data[r[0]:r[-1] + 1] for r in np.split(idx, cuts)]
        x = torch.from_numpy(np.concatenate(parts))
        return x.pin_memory() if self.device.type == "cuda" else x


def _prefetch(make, items, depth=4):
    q = queue.Queue(depth)

    def work():
        for it in items:
            q.put(make(it))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (x := q.get()) is not None:
        yield x


def _cpu_batch(a, src, idx):
    num, name, pieces, y = (torch.from_numpy(a[k][idx]) for k in ("num", "name", "pieces", "y"))
    return num, name, pieces, y, src.get(idx)


def build_model(hp, enc, n_num, fine_tune, pretrained=True):
    bb = resnet18_trunk(pretrained) if fine_tune else None
    return CropTypeNet(n_num, len(enc.names) + 2, len(enc.pieces) + 2, backbone=bb, use_name=hp["use_name"],
                       use_pieces=hp["use_pieces"], emb_dim=hp["emb_dim"], hidden=hp["hidden"],
                       layers=hp["layers"], dropout=hp["dropout"])


def predict(model, a, src, device, batch=4096, chunk=4096, amp=True):
    model.eval()
    out = []
    idxs = [np.arange(i, min(i + chunk, len(a["y"]))) for i in range(0, len(a["y"]), chunk)]
    with torch.no_grad():
        for num, name, pieces, _, img in _prefetch(lambda i: _cpu_batch(a, src, i), idxs):
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                lo = model(num.to(device), name.to(device), pieces.to(device), img.to(device, non_blocking=True))
            out.append(lo.float().softmax(1).cpu())
    return torch.cat(out).numpy()


def train(hp, enc, tr, dv, src_tr, src_dv, dev_screens, seed, device, fine_tune, log=print, pretrained=True):
    """hp: MLP settings (Model 1's) + batch, max_epochs, patience, chunk, and for fine-tuning lr_backbone.
    tr/dv: Encoder.transform arrays. Returns (model with best-epoch weights, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(hp, enc, tr["num"].shape[1], fine_tune, pretrained).to(device)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    groups = [{"params": head, "lr": hp["lr"]}]
    if fine_tune:
        groups.append({"params": model.backbone.parameters(), "lr": hp["lr_backbone"]})
    opt = torch.optim.AdamW(groups, weight_decay=hp["weight_decay"])
    w = torch.tensor(enc.class_weights(hp["class_weight"]), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    amp = hp.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    rng = np.random.default_rng(seed)
    best, best_state, hist, wait = -1.0, None, [], 0
    for ep in range(hp["max_epochs"]):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        batches = chunk_batches(len(tr["y"]), hp["batch"], hp["chunk"], rng)
        for num, name, pieces, y, img in _prefetch(lambda i: _cpu_batch(tr, src_tr, i), batches):
            num, name, pieces, y = num.to(device), name.to(device), pieces.to(device), y.to(device)
            img = img.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(num, name, pieces, img).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
        t1 = time.time()
        p = predict(model, dv, src_dv, device, amp=amp).argmax(1)
        sc = scores(per_screen_confusion(dev_screens, dv["y"], p).sum(0))
        hist.append({"epoch": ep + 1, "train_loss": tot / n, **sc, "train_sec": round(t1 - t0, 1),
                     "dev_sec": round(time.time() - t1, 1)})
        log(f"  ep {ep + 1:2d} loss {tot / n:.4f} dev macro {sc['macro_f1_types']:.4f} "
            f"inv {sc['invalid_f1']:.4f} (train {t1 - t0:.0f}s, dev {time.time() - t1:.0f}s)")
        if sc["macro_f1_types"] > best:
            best, wait = sc["macro_f1_types"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return model, hist
