"""GPU speed test (run on Colab from notebooks/colab_gpu_speed_check.ipynb).

Measures crops per second for a CNN (training step and inference only) at several crop sizes,
plus the CPU time to decode and crop one screenshot. Turns them into hours per pass over the
clean splits. Random inputs only, so the numbers are speeds, not results.

  python scripts/colab_gpu_speed_check.py --config configs/colab_gpu_speed_check.yaml [--quick]
"""
import argparse
import io
import json
import os
import time

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image

from uistruct.runlog import finish_run, start_run


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def gpu_speed(cfg, device, size, amp, train):
    model = getattr(torchvision.models, cfg["model"])(weights=None, num_classes=24).to(device)
    model.train(train)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(enabled=amp and device.type == "cuda")
    bs = cfg["batch_size"]
    x = torch.randn(bs, 3, size, size, device=device)
    y = torch.randint(0, 24, (bs,), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            if train:
                loss = torch.nn.functional.cross_entropy(model(x), y)
            else:
                with torch.no_grad():
                    model(x)
                return
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(cfg["warmup_steps"]):
        step()
    _sync(device)
    t = time.perf_counter()
    for _ in range(cfg["timed_steps"]):
        step()
    _sync(device)
    sec = time.perf_counter() - t
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    return {"crops_per_s": bs * cfg["timed_steps"] / sec, "peak_gpu_gb": mem}


def cpu_speed(cfg):
    """Decode + resize one synthetic screenshot JPEG, then cut and resize crops."""
    w, h = cfg["screenshot_size"]
    rng = np.random.default_rng(cfg["seed"])
    # smooth gradient + mild noise, so the JPEG is closer in size to a real screenshot than pure noise
    base = np.linspace(0, 255, w * h * 3).reshape(h, w, 3)
    arr = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    data = buf.getvalue()
    crop = 128
    t = time.perf_counter()
    for _ in range(cfg["cpu_repeats"]):
        im = Image.open(io.BytesIO(data)).convert("RGB").resize(tuple(cfg["resize_to"]))
        for i in range(cfg["crops_per_screen"]):
            x0, y0 = (i * 20) % 400, (i * 37) % 800
            im.crop((x0, y0, x0 + 100, y0 + 60)).resize((crop, crop))
    sec = (time.perf_counter() - t) / cfg["cpu_repeats"]
    return {"jpeg_bytes": len(data), "seconds_per_screen": sec,
            "crops_per_s_one_core": cfg["crops_per_screen"] / sec, "cpu_cores": os.cpu_count()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/colab_gpu_speed_check.yaml")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a local CPU check")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.quick:
        cfg.update(image_sizes=[64], batch_size=8, warmup_steps=1, timed_steps=2, amp=[False], cpu_repeats=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = start_run("gpu_speed_check" + ("_quick" if args.quick else ""), cfg)

    rows = []
    for size in cfg["image_sizes"]:
        for amp in cfg["amp"]:
            for train in (True, False):
                r = gpu_speed(cfg, device, size, amp, train)
                r.update(size=size, amp=amp, mode="train" if train else "inference")
                rows.append(r)
                print(json.dumps(r))
    cpu = cpu_speed(cfg)
    print(json.dumps(cpu))

    n = cfg["n_elements"]
    total = sum(n.values())
    hours = []
    for r in rows:
        # train: one epoch over train crops; inference: one pass over all crops (frozen-feature option)
        count = n["train"] if r["mode"] == "train" else total
        hours.append({**{k: r[k] for k in ("size", "amp", "mode")},
                      "gpu_hours_per_pass": count / r["crops_per_s"] / 3600})
    cpu_hours = total / (cpu["crops_per_s_one_core"] * cpu["cpu_cores"]) / 3600
    metrics = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
               "torch": torch.__version__, "torchvision": torchvision.__version__,
               "gpu": rows, "cpu": cpu, "hours": hours,
               "cpu_hours_decode_all_crops_all_cores": cpu_hours, "quick": args.quick}
    finish_run(run_dir, metrics)
    print("RESULT_JSON " + json.dumps(metrics))
    print("run folder:", run_dir)


if __name__ == "__main__":
    main()
