import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from r6_analyse import CRIT, cost
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8a84", "#222222", "#6b6b66"
import re as _re
def cl_means(R, k):
    """Average over seeds within each dataset first, so error bars are s.e. over the 15 datasets."""
    d = {}
    for x in R: d.setdefault(_re.sub(r"_s\d+\.npz$", "", x["file"]), []).append(x[k])
    return np.array([np.mean(v) for v in d.values()])
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "pdf.fonttype": 42})
r = json.load(open("r6_analysis.json"))
order = [("holdout1", "holdout", GREY), ("cv2", "2-fold CV", GREY), ("cv5", "5-fold CV", GREY), ("cv5_bc", "5-fold CV, noise-corrected", BLUE),
         ("lf2", "label-free, 2 fold models", ORANGE), ("lf5", "label-free, 5 fold models", ORANGE), ("switch5", "switch rule, 5 fold models", ORANGE),
         ("V5", "V alone, 5 fold models", ORANGE)]
spec = {k: s for k, _, s in CRIT}
relc = {k: np.median([cost(x, spec[k]) / x["cost_holdout1"] for x in r]) for k, _, _ in order}
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
for ax, (g, lab) in zip(axs, [("sym0.2", "symmetric noise"), ("classdep", "class-dependent noise"), ("instdep", "instance-dependent noise")]):
    R = [x for x in r if g in x["file"]]
    for i, (k, name, col) in enumerate(order):
        v = cl_means(R, "reg_" + k); m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar([m], [i], xerr=[se], fmt="o", color=col, ms=5, capsize=2)
    ax.invert_yaxis(); ax.set_title(f"{lab}", fontsize=9); ax.set_xlabel("selection regret (clean test error)")
    ax.set_xlim(0, None)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([f"{n}  ({relc[k]:.1f}x)" for k, n, _ in order])
fig.suptitle("Mean regret ± s.e. over 15 datasets (5 seeds each); compute relative to one holdout fit in brackets", fontsize=8.5, color=MUTED, y=1.0)
fig.tight_layout(); fig.savefig("paper/figs/r6_cost.pdf", bbox_inches="tight"); fig.savefig("paper/figs/r6_cost.png", dpi=140, bbox_inches="tight"); print("ok")
