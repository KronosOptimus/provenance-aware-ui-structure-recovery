"""Shared style for report figures: every figure in colour, using bright, light, peaceful
tones with light and shade. Each hue has a soft tone (fills) and a deep shade (the bar that matters, lines,
outlines). Fixed meanings across the report:
  SKY       main series / valid elements / neural MLP rows
  PEACH     invalid (BACKGROUND) elements / trees rows / second series
  MINT      non-learned rule baselines / third series
  LAVENDER, ROSE   spare hues for later figures
  GREY      neutral context (ROOT, whiskers, reference lines)
"""
import matplotlib.pyplot as plt

SOFT = {
    "sky":      ("#a8d4f5", "#4a90d9"),
    "peach":    ("#ffc9a8", "#f08a5d"),
    "mint":     ("#b5e8cf", "#4fb58a"),
    "lavender": ("#d4c8f0", "#8c74cf"),
    "rose":     ("#f7c1cf", "#d9668a"),
}
(SKY, SKY_D), (PEACH, PEACH_D), (MINT, MINT_D) = SOFT["sky"], SOFT["peach"], SOFT["mint"]
(LAVENDER, LAVENDER_D), (ROSE, ROSE_D) = SOFT["lavender"], SOFT["rose"]
GREY, INK = "#a7a6a1", "#333333"


def use_report_style():
    """Serif text to match the Times New Roman body; recessive axes."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#666666", "savefig.dpi": 300, "savefig.bbox": "tight"})
