"""Demo checks and demo screen choice. Laptop CPU, no test-set choices are made here.

  --stage check   the demo pipeline (src/uistruct/demo.py) on randomly drawn test screens at every level,
                  compared with the per-screen rows of the provenance test run (same system: fusion_full_s0 with trust
                  removal; at the absent level the "nothing removed" row, plan section 7)
  --stage pick    demo screens by a rule fixed before looking at them, read from the
                  per-screen rows of the provenance test run

  python scripts/demo_checks_and_screen_choice.py --stage check [--n 200]
  python scripts/demo_checks_and_screen_choice.py --stage pick
"""
import argparse
import time

import numpy as np
import pandas as pd

from uistruct.demo import LEVELS, SYSTEM, Demo
from uistruct.runlog import finish_run, start_run

PER_SCREEN = "results/runs/colab_trust_score/trust_score_test_run1/per_screen.csv.gz"
COLS = ["screen_id", "system", "level", "truth", "n_elements", "n_removed", "removed_valid", "kept_unfaithful",
        "container_f1", "n_true_containers", "n_pred_containers"]


def test_rows():
    df = pd.read_csv(PER_SCREEN, usecols=COLS)
    df = df[(df.truth == "raw") & df.system.isin([SYSTEM, "fusion_full_s0"])]
    sys_at = lambda lv: "fusion_full_s0" if lv == "absent" else SYSTEM  # noqa: E731
    return df[df.system == df.level.map(sys_at)].set_index(["screen_id", "level"])


def stage_check(a):
    ref = test_rows()
    sids = sorted(ref.index.get_level_values(0).unique())
    pick = sorted(np.random.default_rng(a.seed).choice(sids, a.n, replace=False).tolist())
    d = Demo()
    d.load(pick)
    rows, t0 = [], time.time()
    for s in pick:
        for lv in LEVELS:
            r = d.run(s, lv)
            t = ref.loc[(s, lv)]
            rows.append({"screen_id": s, "level": lv, "f1_demo": r["answer"]["container_f1"],
                         "f1_test": round(float(t.container_f1), 3),
                         "removed_demo": r["summary"]["removed"], "removed_test": int(t.n_removed) if lv != "absent" else 0,
                         "pred_demo": r["summary"]["containers"], "pred_test": int(t.n_pred_containers),
                         "elements_demo": r["summary"]["elements"], "elements_test": int(t.n_elements),
                         "sec": r["seconds"]})
    df = pd.DataFrame(rows)
    df["f1_diff"] = (df.f1_demo - df.f1_test).abs()
    out = {"n_screens": a.n, "seed": a.seed, "levels": LEVELS, "minutes": round((time.time() - t0) / 60, 1),
           "sec_per_run": {"median": float(df.sec.median()), "max": float(df.sec.max())},
           "note": "f1 = container F1 (raw truth, empty-screen rule) rounded to 3 decimals; a difference of 0.001 is rounding"}
    for lv, x in df.groupby("level"):
        out[lv] = {"same_elements": float((x.elements_demo == x.elements_test).mean()),
                   "same_removed": float((x.removed_demo == x.removed_test).mean()),
                   "same_pred_containers": float((x.pred_demo == x.pred_test).mean()),
                   "f1_within_0.001": float((x.f1_diff <= 0.0011).mean()), "f1_max_diff": float(x.f1_diff.max()),
                   "f1_mean_demo": float(x.f1_demo.mean()), "f1_mean_test": float(x.f1_test.mean())}
    return out, df


def stage_pick(a):
    """Rule (fixed before looking): test screens in portrait with 8-30 elements at the real level
    and at least one true container; system fusion_full_s0 + trust, raw truth.
      showcase (3): real-level container F1 >= 0.95, at least one unfaithful element removed, no valid element
                    removed; ranked by heavy-level container F1 (high first), then fewer unfaithful elements kept
                    at the real level, then screen id
      typical (1):  real-level container F1 closest to the test mean of the system; ties by screen id
      failure (1):  lowest real-level container F1; ties by screen id
    A screen whose screenshot is blank or unreadable is skipped by eye and the next in rank taken (logged)."""
    ref = test_rows().reset_index()
    real = ref[ref.level == "real"].set_index("screen_id")
    heavy = ref[ref.level == "heavy"].set_index("screen_id")
    scr = pd.read_parquet("data/processed/screens.parquet", columns=["screen_id", "img_w", "img_h"]).set_index("screen_id")
    ok = real[(real.n_elements.between(8, 30)) & (real.n_true_containers > 0)]
    ok = ok[scr.loc[ok.index, "img_h"].to_numpy() > scr.loc[ok.index, "img_w"].to_numpy()]
    rem_unf = ok.n_removed - ok.removed_valid
    show = ok[(ok.container_f1 >= 0.95) & (rem_unf >= 1) & (ok.removed_valid == 0)].copy()
    show["heavy_f1"] = heavy.loc[show.index, "container_f1"]
    show = show.reset_index().sort_values(["heavy_f1", "kept_unfaithful", "screen_id"], ascending=[False, True, True])
    mean = float(real.container_f1.mean())
    typ = ok.reset_index().assign(gap=lambda x: (x.container_f1 - mean).abs()).sort_values(["gap", "screen_id"])
    fail = ok.reset_index().sort_values(["container_f1", "screen_id"])
    keep = ["screen_id", "n_elements", "container_f1", "n_removed", "removed_valid", "kept_unfaithful", "n_true_containers"]
    return {"rule": stage_pick.__doc__.strip().splitlines(), "eligible": int(len(ok)), "test_mean_real": mean,
            "showcase_ranked": show[keep + ["heavy_f1"]].head(10).to_dict("records"),
            "typical_ranked": typ[keep].head(5).to_dict("records"),
            "failure_ranked": fail[keep].head(5).to_dict("records")}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "pick"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run_dir = start_run(f"demo_{a.stage}", {"stage": a.stage, "n": a.n, "seed": a.seed, "per_screen": PER_SCREEN,
                                          "system": SYSTEM})
    res, df = {"check": stage_check, "pick": stage_pick}[a.stage](a)
    if df is not None:
        df.to_csv(run_dir / "per_screen.csv", index=False)
    finish_run(run_dir, res)
    print({k: v for k, v in res.items() if k not in ("showcase_ranked", "typical_ranked", "failure_ranked")})
    print("run dir:", run_dir)


if __name__ == "__main__":
    main()
