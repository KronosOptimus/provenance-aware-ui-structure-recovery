"""Model 1 report figure: (a) type head on clean test, macro-F1 over 23 types with 95% CIs; (b) learning curve on
clean dev (nested company subsets). Reads the run folders only; no computation.
Run: python scripts/plot_element_types_metadata_mlp.py
"""
import json

import matplotlib.pyplot as plt

from uistruct.plotting import INK, MINT, MINT_D, PEACH, PEACH_D, SKY, SKY_D, use_report_style

TEST = "results/runs/element_types_mlp_test_run1/metrics.json"
DEV = "results/runs/element_types_mlp_dev_run1/metrics.json"
OUT = "results/figures/element_types_metadata_mlp.png"

use_report_style()
COL = {"majority": (MINT, MINT_D), "lookup": (MINT, MINT_D), "gbdt": (PEACH, PEACH_D)}   # rules mint, trees peach, MLPs sky

te, dv = json.load(open(TEST)), json.load(open(DEV))
rows = [("Majority", "majority"), ("Class-name lookup", "lookup"), ("MLP: box", "box_s0"),
        ("MLP: + tree, flags", "box_tree_flags_s0"), ("Trees: Model 1 inputs", "gbdt"), ("MLP: Model 1", "model1_s0")]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axs[0]
for i, (lab, k) in enumerate(rows):
    m = te["test"][k]["macro_f1_types"]
    soft, deep = COL.get(k, (SKY, SKY_D))
    ax.barh(i, m["value"], color=deep if k == "model1_s0" else soft, edgecolor=deep, lw=0.6)
    ax.errorbar(m["value"], i, xerr=[[m["value"] - m["lo"]], [m["hi"] - m["value"]]], color=INK, capsize=2, lw=0.8)
    ax.text(m["hi"] + 0.01, i, f"{m['value']:.3f}", va="center", fontsize=7)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 0.85)
ax.set_xlabel("Test macro-F1, 23 types (95% CI)")
ax.set_title("(a) Type head, clean test", fontsize=9)

ax = axs[1]
lc = dv["dev_learning_curve"]
xs = [float(k) * 100 for k in lc]
ax.plot(xs, list(lc.values()), "o-", color=SKY_D, mfc=SKY, mec=SKY_D, ms=6, lw=2)
ax.axhline(dv["dev"]["lookup"]["macro_f1_types"]["value"], color=MINT_D, ls="--", lw=1.2)
ax.text(27, dv["dev"]["lookup"]["macro_f1_types"]["value"] + 0.006, "lookup", fontsize=7, color=INK)
ax.set_xlabel("% of clean train screens\n(grown by whole companies)")
ax.set_ylabel("Dev macro-F1")
ax.set_xticks([25, 50, 75, 100])
ax.set_title("(b) Model 1 learning curve, dev", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
