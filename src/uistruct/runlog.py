"""Run logging. Every experiment calls `start_run` and `finish_run`.

Each run gets results/runs/<name>_run<N>/ (N = next free number) with:
  config.yaml   the exact config used
  env.json      python/platform/package versions, git commit and whether the tree was dirty, seed
  metrics.json  written by finish_run
Small result files are committed; checkpoints (*.pt) are git-ignored.
"""
import json
import platform
import random
import re
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

PACKAGES = ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "pillow"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def environment() -> dict:
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            vers[p] = None
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return {"python": platform.python_version(), "platform": platform.platform(), "gpu": gpu,
            "packages": vers, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain"))}


def start_run(name: str, cfg: dict, root="results/runs") -> Path:
    env = environment()  # before creating the run folder, so the folder itself doesn't count as "dirty"
    taken = [int(m.group(1)) for p in Path(root).glob(f"{name}_run*")
             if (m := re.fullmatch(rf"{re.escape(name)}_run(\d+)", p.name))]
    run_dir = Path(root) / f"{name}_run{max(taken, default=0) + 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    env["seed"] = cfg.get("seed")
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    if cfg.get("seed") is not None:
        set_seed(cfg["seed"])
    return run_dir


def finish_run(run_dir: Path, metrics: dict):
    (Path(run_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
