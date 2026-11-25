"""Model 2 report figure: (a) clean test, type macro-F1 and invalid AUROC with 95% CIs for the main rows;
(b) per-class test F1 change, Model 2 - Model 1 (seed 0). Reads the run folder only; no computation.
Run: python scripts/plot_element_types_with_colour.py
"""
import json

import matplotlib.pyplot as plt

from uistruct.plotting import INK, PEACH, PEACH_D, SKY, SKY_D, use_report_style

TEST = "results/runs/element_types_colour_test_run1/metrics.json"
OUT = "results/figures/element_types_with_colour.png"

use_report_style()

te = json.load(open(TEST))
rows = [("MLP: box", "box_s0"), ("MLP: box + colour", "box_colour_s0"), ("Trees: Model 1 inputs", "gbdt"),
        ("Trees: Model 2 inputs", "gbdt2"), ("MLP: Model 1", "model1_s0"), ("MLP: Model 2", "model2_s0")]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axs[0]
h = 0.38
for i, (lab, k) in enumerate(rows):
    for j, (met, (soft, deep)) in enumerate([("macro_f1_types", (SKY, SKY_D)), ("invalid_auroc", (PEACH, PEACH_D))]):
        m = te["scores"][k][met]
        y = i + (h / 2 if j == 0 else -h / 2)
        ax.barh(y, m["value"], height=h, color=deep if k == "model2_s0" else soft, edgecolor=deep, lw=0.6, label=None if i else ["Macro-F1, 23 types", "Invalid AUROC"][j])
        ax.errorbar(m["value"], y, xerr=[[m["value"] - m["lo"]], [m["hi"] - m["value"]]], color=INK, capsize=1.5, lw=0.7)
        ax.text(m["hi"] + 0.01, y, f"{m['value']:.3f}", va="center", fontsize=6.5)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 1.05)
ax.set_xlabel("Clean test (95% CI over screens)")
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2)
ax.set_title("(a) Type head with and without colour", fontsize=9)

ax = axs[1]
pc = sorted(((n, v["delta"]) for n, v in te["per_class_f1"].items() if v["delta"] is not None), key=lambda t: t[1])
ax.barh(range(len(pc)), [d for _, d in pc], color=[SKY if d >= 0 else PEACH for _, d in pc], edgecolor=[SKY_D if d >= 0 else PEACH_D for _, d in pc], lw=0.6)
ax.set_yticks(range(len(pc)))
ax.set_yticklabels([n.replace("INVALID(BACKGROUND)", "INVALID").replace("_", " ").lower() for n, _ in pc], fontsize=6)
ax.axvline(0, color=INK, lw=0.6)
ax.set_xlabel("F1 change, Model 2 - Model 1 (test)")
ax.set_title("(b) Per class", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT, "from", TEST)
