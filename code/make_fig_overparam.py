import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8a84", "#222222", "#6b6b66"
import re as _re
def cl_means(R, k):
    """Average over seeds within each dataset first, so error bars are s.e. over the 15 datasets."""
    d = {}
    for x in R: d.setdefault(_re.sub(r"_s\d+\.npz$", "", x["file"]), []).append(x[k])
    return np.array([np.mean(v) for v in d.values()])
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "pdf.fonttype": 42})
r = json.load(open("r6_overparam.json"))
cost = {"holdout1": lambda x: 1.0, "cv2": lambda x: x["cost_cv2"] / x["cost_holdout1"], "cv5": lambda x: x["cost_cv5"] / x["cost_holdout1"],
        "cv5_bc": lambda x: x["cost_poolfit5"] / x["cost_holdout1"], "V2": lambda x: (x["cost_cv2"] + x["cost_unlab2"]) / x["cost_holdout1"],
        "V5": lambda x: (x["cost_cv5"] + x["cost_unlab5"]) / x["cost_holdout1"], "lf2": lambda x: x["cost_pool2"] / x["cost_holdout1"], "lf5": lambda x: x["cost_pool5"] / x["cost_holdout1"]}
order = [("holdout1", "holdout (labels)", GREY), ("cv2", "2-fold CV (labels)", GREY), ("cv5", "5-fold CV (labels)", GREY),
         ("cv5_bc", "5-fold CV, noise-corrected (labels)", GREY), ("V2", "min V, 2 disjoint halves (no labels)", BLUE),
         ("V5", "min V, 5 folds (no labels)", BLUE), ("lf2", "consensus + V, 2 halves (no labels)", BLUE), ("lf5", "consensus + V, 5 folds (no labels)", BLUE)]
fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.0), sharey=True)
for ax, (g, lab) in zip(axs, [("sym0.2", "symmetric noise"), ("classdep", "class-dependent noise"), ("instdep", "instance-dependent noise")]):
    R = [x for x in r if g in x["file"]]
    for i, (k, name, col) in enumerate(order):
        v = cl_means(R, "reg_" + k); ax.errorbar([v.mean()], [i], xerr=[v.std(ddof=1) / np.sqrt(len(v))], fmt="o", color=col, ms=5, capsize=2)
    ax.invert_yaxis(); ax.set_title(lab, fontsize=9); ax.set_xlabel("selection regret (clean test error)"); ax.set_xlim(0, None)
rel = {k: np.median([cost[k](x) for x in r]) for k, _, _ in order}
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([f"{n}  [{rel[k]:.1f}x]" for k, n, _ in order])
fig.suptitle("Choosing among 6 overparameterized learners on one noisy dataset (15 datasets x 5 seeds per panel); mean ± s.e. over datasets; [compute vs one holdout fit]", fontsize=8.2, color=MUTED, y=1.01)
fig.tight_layout(); fig.savefig("paper/figs/r6_overparam.pdf", bbox_inches="tight"); fig.savefig("paper/figs/r6_overparam.png", dpi=140, bbox_inches="tight"); print("ok")
