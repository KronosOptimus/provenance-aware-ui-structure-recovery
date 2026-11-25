"""System diagram of the implemented pipeline (data -> features -> type head -> evaluation).

  python scripts/plot_system_overview.py   ->  results/figures/system_overview.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from uistruct.plotting import LAVENDER, LAVENDER_D, MINT, MINT_D, PEACH, PEACH_D, ROSE, ROSE_D, SKY, SKY_D

plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, title, lines, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge, lw=1.3))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=10.5, weight="bold", color="#2b2b2b")
    ax.text(x + w / 2, y + h - 5.0, "\n".join(lines), ha="center", va="top", fontsize=8.6, color="#333333",
            linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#6b6b6b"))


# column 1: data
box(1, 26, 18, 18, "RICO", ["screenshots", "raw view hierarchies", "6.18M nodes"], SKY, SKY_D)
box(1, 2, 18, 18, "CLAY", ["cleaned labels", "23 types + invalid", "59,555 screens"], SKY, SKY_D)
# column 2: preparation
box(24, 14, 19, 30, "Data preparation", ["join on node id", "(1,427,913 nodes)", "", "leakage-free splits",
                                         "by company and layout", "", "resize to 540 x 960",
                                         "fit on train only"], LAVENDER, LAVENDER_D)
# column 3: features
box(48, 32, 22, 12, "Model 1 inputs", ["box, raw tree position,", "flags, class name"], MINT, MINT_D)
box(48, 17.5, 22, 12, "Model 2: + colour", ["18 crop statistics"], MINT, MINT_D)
box(48, 2, 22, 13, "Model 3: + CNN", ["96 px crop + margin", "ResNet-18 (ImageNet),", "frozen or fine-tuned"],
    MINT, MINT_D)
# column 4: head and evaluation
box(75, 26, 24, 18, "Type head (MLP)", ["element type", "(23 classes)", "invalid / valid"], PEACH, PEACH_D)
box(75, 2, 24, 19, "Evaluation", ["macro-F1, invalid F1,", "invalid AUROC", "bootstrap CIs, paired tests",
                                  "trees + lookup baselines"], ROSE, ROSE_D)

arrow(19.8, 35, 23.6, 31)
arrow(19.8, 11, 23.6, 22)
for y in (38, 23.5, 8.5):
    arrow(43.8, 29, 47.6, y)
for y in (38, 23.5, 8.5):
    arrow(70.8, y, 74.6, 35)
arrow(87, 25.4, 87, 21.6)
fig.savefig("results/figures/system_overview.png", dpi=220, bbox_inches="tight", facecolor="white")
print("saved")
