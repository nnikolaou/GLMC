import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9a94", "#222222", "#6b6b66"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
    "lines.linewidth": 1.8, "legend.frameon": False, "pdf.fonttype": 42})
r = json.load(open("sc_analysis.json"))
def series(cond, key, xs, xkey):
    m, se = [], []
    for x in xs:
        v = [q[key] for q in r if cond(q) and abs(q[xkey] - x) < 1e-9]
        # average pools within a seed first, then mean and s.e. over seeds
        by = {}
        for q in [q for q in r if cond(q) and abs(q[xkey] - x) < 1e-9]: by.setdefault(q["file"], []).append(q[key])
        s = np.array([np.mean(b) for b in by.values()]); m.append(s.mean()); se.append(s.std(ddof=1) / np.sqrt(len(s)))
    return np.array(m), np.array(se)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
shares = sorted({q["linear_share"] for q in r if q["f"] == 0}); c0 = lambda q: q["f"] == 0
for key, col, ls, lab in [("holdout", GREY, "--", "holdout, 300 noisy labels"), ("holdout40", GREY, "-", "holdout, 40 noisy labels"),
                          ("labelfree", ORANGE, "-", "label-free, plain consensus"), ("labelfree_adaptive", BLUE, "-", "label-free, adaptive clustering")]:
    m, se = series(c0, key, shares, "linear_share")
    axs[0].errorbar(shares, m, yerr=se, color=col, ls=ls, marker="o", ms=3, capsize=2, label=lab)
axs[0].set_xlabel("share of candidates from the shared-bias (linear) family"); axs[0].set_ylabel("selection regret (excess risk)")
axs[0].set_title("Shared bias from redundant candidates", fontsize=9); axs[0].legend(fontsize=6.5, loc="upper left")
fs = sorted({q["f"] for q in r}); c1 = lambda q: q["linear_share"] == 0
dn = [np.mean([q["delta_noise"] for q in r if q["f"] == f and c1(q)]) for f in fs]
for key, col, ls, lab in [("holdout", GREY, "-", "holdout, noisy labels"), ("holdout_bc", BLUE, "--", "noise-corrected holdout"),
                          ("labelfree", ORANGE, "-", "label-free")]:
    m, se = series(c1, key, fs, "f")
    axs[1].errorbar(fs, m, yerr=se, color=col, ls=ls, marker="o", ms=3, capsize=2, label=lab)
axs[1].set_xticks(fs); axs[1].set_xticklabels([f"{f:.1f}\n$\\delta_T$={d:.2f}" for f, d in zip(fs, dn)])
axs[1].set_xlabel("class-dependent flip rate $0\\to1$"); axs[1].set_title("Shared bias from asymmetric label noise", fontsize=9)
axs[1].legend(fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig("paper/figs/s_consensus.pdf", bbox_inches="tight"); fig.savefig("paper/figs/s_consensus.png", dpi=140, bbox_inches="tight")
print("ok")
