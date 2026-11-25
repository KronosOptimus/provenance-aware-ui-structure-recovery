"""Provenance-network analysis on the laptop, from the Colab run folders only (no model, no new fit, nothing selected on test).

Reads results/runs/colab_trust_score/{trust_score_dev_run2, trust_score_test_run1, structure_leftover_test_run1}. Writes a run folder with
  - comparisons added after the validation results (labelled post hoc in the report):
      * removal by the no-head network's type head vs the full model's trust removal, per level;
      * the provenance network at the absent level vs Model 5 on the same screens (the specialist without metadata);
      * the provenance network at the real level vs Model 4 (Model 4 is given the valid leaves only, so this is the cost of the
        invalid elements real metadata delivers, not a like-for-like model comparison);
      * the absent level under plan §7 (no removal there; the scoring code removed a few elements)
  - figures results/figures/trust_score_levels.png and trust_score_reliability.png.
Run: python scripts/analyse_trust_score_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uistruct.eval.stats import paired_bootstrap
from uistruct.plotting import (GREY, INK, LAVENDER, LAVENDER_D, MINT_D, PEACH, PEACH_D, ROSE, ROSE_D, SKY, SKY_D,
                               use_report_style)
from uistruct.runlog import finish_run, start_run

RUNS = Path("results/runs/colab_trust_score")
LEVELS = ["real", "inject2", "heavy", "drop50", "absent"]
LEVEL_NAMES = {"real": "real", "inject2": "fakes x2", "heavy": "heavy", "drop50": "drop 50%", "absent": "absent"}
CFG = {"seed": 0, "n_boot": 1000, "dev": str(RUNS / "trust_score_dev_run2"), "test": str(RUNS / "trust_score_test_run1"),
       "structure_test": str(RUNS / "structure_leftover_test_run1")}


def cf1_rule(df):
    """Empty-screen rule: no true and no predicted container = 1.0 (provenance files already apply it; Models 4-6 files are checked)."""
    df = df.copy()
    both = (df.n_true_containers == 0) & (df.n_pred_containers == 0)
    df.loc[both, "container_f1"] = 1.0
    return df


def wide(df, truth="raw"):
    x = df[df.truth == truth]
    return x.pivot_table(index="screen_id", columns=["system", "level"], values="container_f1")


def pair(d, a, b, n_boot, seed):
    r = paired_bootstrap(d[a].to_numpy(), d[b].to_numpy(), n_boot, seed)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}


def main():
    run = start_run("trust_score_analysis", CFG)
    out = {"note": "post hoc (after the validation results); nothing here was used for any choice"}
    for split in ("dev", "test"):
        df = cf1_rule(pd.read_csv(Path(CFG[split]) / "per_screen.csv.gz"))
        res = {}
        for truth in ("raw", "smoothed"):
            w = wide(df, truth)
            for lv in LEVELS:
                a, b = ("fusion_nohead_s0+typehead", lv), ("fusion_full_s0+trust", lv)
                res[f"nohead+typehead_minus_full+trust|{lv}|{truth}"] = pair(w, a, b, CFG["n_boot"], CFG["seed"])
                a = ("fusion_nohead_s0+typehead", lv)
                res[f"nohead+typehead_minus_nohead|{lv}|{truth}"] = pair(w, a, ("fusion_nohead_s0", lv), CFG["n_boot"], CFG["seed"])
        ab = df[(df.level == "absent") & (df.truth == "raw")]
        res["absent_removed_per_screen"] = {s: {"mean_removed": round(float(x.n_removed.mean()), 4),
                                                "screens_with_removal": int((x.n_removed > 0).sum())}
                                            for s, x in ab.groupby("system") if x.n_removed.sum() > 0}
        out[split] = res
        if split == "test":
            test_df = df

    # Model 4 / Model 5 on the same test screens
    t9 = cf1_rule(pd.read_csv(Path(CFG["structure_test"]) / "per_screen.csv.gz"))
    rows = {}
    for truth in ("raw", "smoothed"):
        a = t9[t9.truth == truth].pivot(index="screen_id", columns="system", values="container_f1")
        a["model5_seedmean"] = a[["model5", "model5_s1", "model5_s2"]].mean(1)
        a["model4_seedmean"] = a[["model4", "model4_s1", "model4_s2"]].mean(1)
        t = wide(test_df, truth)
        both = a.index.intersection(t.index)
        assert len(both) == len(a) == len(t), "Models 4-6 and provenance test screens differ"
        a, t = a.loc[both], t.loc[both]
        for m in ("model5", "model5_seedmean"):
            for s in ("fusion_full_s0", "fusion_nohead_s0"):
                rows[f"{s}@absent_minus_{m}|{truth}"] = pair(pd.DataFrame({"x": t[(s, "absent")], "y": a[m]}), "x", "y",
                                                             CFG["n_boot"], CFG["seed"])
        for s in ("fusion_full_s0", "fusion_full_s0+trust", "fusion_full_s0+oracle", "fusion_nohead_s0+typehead"):
            rows[f"{s}@real_minus_model4|{truth}"] = pair(pd.DataFrame({"x": t[(s, "real")], "y": a["model4"]}), "x", "y",
                                                          CFG["n_boot"], CFG["seed"])
        rows[f"means|{truth}"] = {k: round(float(a[k].mean()), 4) for k in ("model4", "model5", "model4_seedmean", "model5_seedmean")}
    out["vs_structure_test"] = rows
    finish_run(run, out)
    figures(test_df)
    print("saved", run)


def figures(df):
    use_report_style()
    te = json.load(open(Path(CFG["test"]) / "metrics.json"))
    st = te["structure"]
    rows = [("Nothing removed", "fusion_full_s0", GREY, GREY, "o", "-"),
            ("Trust removal", "fusion_full_s0+trust", SKY, SKY_D, "o", "-"),
            ("Oracle removal", "fusion_full_s0+oracle", MINT_D, MINT_D, "^", ":"),
            ("No trust head, type-head removal", "fusion_nohead_s0+typehead", PEACH, PEACH_D, "s", "-"),
            ("Trained on real metadata only", "fusion_realonly_s0+trust", LAVENDER, LAVENDER_D, "D", "--"),
            ("Trained without fakes", "fusion_noinject_s0+trust", ROSE, ROSE_D, "v", "--")]
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    x = np.arange(len(LEVELS))
    for i, (lab, s, soft, deep, mk, ls) in enumerate(rows):
        v = [st[f"{s}|{lv}|raw"]["container_f1"] for lv in LEVELS]
        m = np.array([r["mean"] for r in v])
        err = np.array([[r["mean"] - r["lo"] for r in v], [r["hi"] - r["mean"] for r in v]])
        off = (i - 2.5) * 0.045
        ax.errorbar(x + off, m, yerr=err, color=deep, mfc=soft, mec=deep, marker=mk, ms=4.5, lw=1.1, ls=ls,
                    capsize=1.5, label=lab)
    t9 = json.load(open(Path(CFG["structure_test"]) / "metrics.json"))
    m4 = t9["summary"]["model4|raw"]["container_f1"]["mean"]
    m5 = t9["summary"]["model5|raw"]["container_f1"]["mean"]
    ax.hlines(m4, -0.3, 0.3, color=INK, lw=0.8, ls="-.")
    ax.text(0.32, m4, f"Model 4, valid leaves only ({m4:.3f})", fontsize=6.5, va="center")
    ax.hlines(m5, 3.7, 4.3, color=INK, lw=0.8, ls="-.")
    ax.text(4.32, m5, f"Model 5\n({m5:.3f})", fontsize=6.5, va="center")
    ax.set_xlim(-0.4, 4.75)
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_NAMES[lv] for lv in LEVELS])
    ax.set_xlabel("Metadata condition (clean test)")
    ax.set_ylabel("Container F1, raw truth (95% CI)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig("results/figures/trust_score_levels.png")
    plt.close(fig)

    tr = te["trust"]["sources"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=":")
    for lab, k, kind, col, mk in (("Trust head, raw", "fusion_full_s0:trust", "reliability_raw", SKY_D, "o"),
                                  ("Trust head, temperature", "fusion_full_s0:trust", "reliability_temperature", LAVENDER_D, "s"),
                                  ("Trees on Model 2 inputs, raw", "gbdt2", "reliability_raw", PEACH_D, "^")):
        b = [r for r in tr[k][kind] if r["n"] >= 30]
        ax.plot([1 - r["conf"] for r in b], [1 - r["acc"] for r in b], marker=mk, ms=3.5, lw=1, color=col, label=lab)
    ax.set_xlabel("Predicted p(unfaithful)")
    ax.set_ylabel("Observed share unfaithful")
    ax.set_title("(a) Reliability, clean test (bins with ≥ 30 elements)", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax = axs[1]
    src = [("Trust head", "fusion_full_s0:trust", SKY, SKY_D), ("Type head, same network", "fusion_full_s0:typehead", LAVENDER, LAVENDER_D),
           ("Box + CNN", "boxcnn", ROSE, ROSE_D), ("Model 3", "model3", ROSE, ROSE_D),
           ("Trees, Model 2 inputs", "gbdt2", PEACH, PEACH_D), ("Box + colour", "box_colour", PEACH, PEACH_D),
           ("Model 2", "model2", PEACH, PEACH_D)]
    for i, (lab, k, soft, deep) in enumerate(src):
        c = tr[k]["auroc_ci"]
        ax.barh(i, c["value"], color=soft, edgecolor=deep, lw=0.6, height=0.6)
        ax.errorbar(c["value"], i, xerr=[[c["value"] - c["lo"]], [c["hi"] - c["value"]]], color=INK, capsize=1.5, lw=0.7)
        ax.text(c["hi"] + 0.004, i, f"{c['value']:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(src)))
    ax.set_yticklabels([s[0] for s in src], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("AUROC, unfaithful element = positive (95% CI)")
    ax.set_title("(b) Finding unfaithful elements, real level", fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/figures/trust_score_reliability.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
