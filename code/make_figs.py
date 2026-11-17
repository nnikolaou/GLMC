import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9a94", "#222222", "#6b6b66"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "legend.frameon": False, "pdf.fonttype": 42})
D = "paper/figs/"
e1 = json.load(open("e1_results.json"))

# Fig 1a: V vs n for interpolators at eps=0.2
fig, ax = plt.subplots(figsize=(3.4, 2.6))
names = {"MemSmooth (interp.)": ("Memorize-and-smooth", BLUE, "-"), "Random forest": ("Random forest", BLUE, "--"),
         "GBoost (exp, lr=1, d=6)": ("Boosting (lr 1)", GREY, "-."), "1-NN": ("1-NN", GREY, ":"),
         "Tree (full depth)": ("Full tree", GREY, "--"), "MLP 512 (interp.)": ("MLP 512", ORANGE, "-")}
HYX = [x for x in e1 if x["eps"] == 0.2][0]["H_YgX"]
for k, (lab, c, ls) in names.items():
    pts = sorted([(x["n"], x["V"]) for x in e1 if x["eps"] == 0.2 and x["learner"] == k])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, marker="o", ms=4, label=lab)
ax.axhline(HYX, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(600, HYX - 0.05, f"stored noise H(Y|X) = {HYX:.2f} bits", color=MUTED, fontsize=7.5)
ax.set_xscale("log", base=2); ax.set_xlabel("training-set size n"); ax.set_ylabel("expressed information V (bits)")
ax.set_ylim(0, 0.95); ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
fig.savefig(D + "e1_V_vs_n.pdf", bbox_inches="tight"); fig.savefig(D + "e1_V_vs_n.png", dpi=150, bbox_inches="tight")

# Fig 1b (left panel of Fig. 1): drawn by make_fig_e1_band.py (two-sided band of Theorem 9)

# Fig 2: boosting trajectories
e5 = [x for x in json.load(open("e5_results.json")) if x["eps"] == 0.2 and x["lr"] == 0.1]
r = [x["round"] for x in e5]
best = min(e5, key=lambda x: x["excess_risk"])["round"]; interp_r = next(x["round"] for x in e5 if x["train_err"] < 0.005)
fig, axs = plt.subplots(3, 1, figsize=(6, 5.2), sharex=True)
for ax, key, lab, c in zip(axs, ["leg_IFX", "V", "excess_risk"],
                           ["old plane: $I_S(F;X)/\\log n$", "expressed info V (bits)", "excess 0-1 risk"], [GREY, BLUE, BLUE]):
    ax.plot(r, [x[key] for x in e5], color=c); ax.set_ylabel(lab, fontsize=8); ax.set_ylim(0, None)
    for xv in (best, interp_r): ax.axvline(xv, color=MUTED, lw=0.8, ls=(0, (4, 3)))
axs[0].text(best + 2, axs[0].get_ylim()[1] * 0.1, f"best test round {best}", color=MUTED, fontsize=7.5)
axs[0].text(interp_r + 2, axs[0].get_ylim()[1] * 0.1, f"train error < 0.5% (round {interp_r})", color=MUTED, fontsize=7.5)
axs[-1].set_xlabel("boosting round")
fig.savefig(D + "e5_boost.pdf", bbox_inches="tight"); fig.savefig(D + "e5_boost.png", dpi=150, bbox_inches="tight")

# Fig 3: double descent
e2 = json.load(open("e2_results.json"))
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
for lam, c, lab in [(0.0, BLUE, "min-norm (interpolates for p ≥ n)"), (0.01, GREY, "ridge λ = 0.01")]:
    pts = sorted([x for x in e2 if x["eps"] == 0.2 and x["lam"] == lam], key=lambda x: x["p"])
    for ax, key in zip(axs, ["excess_risk", "V"]):
        ax.plot([x["ratio"] for x in pts], [x[key] for x in pts], color=c, marker="o", ms=3, label=lab)
for ax, lab in zip(axs, ["excess 0-1 risk", "expressed information V (bits)"]):
    ax.set_xscale("log", base=2); ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (4, 3))); ax.set_xlabel("width p / n"); ax.set_ylabel(lab); ax.set_ylim(0, None)
axs[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(D + "e2_dd.pdf", bbox_inches="tight"); fig.savefig(D + "e2_dd.png", dpi=150, bbox_inches="tight")

# Fig 4: estimator
e3 = json.load(open("e3_results.json"))
fig, ax = plt.subplots(figsize=(3.8, 2.6))
for name, c in [("1-NN", GREY), ("Full-depth tree", MUTED), ("Random forest", BLUE)]:
    pts = [x for x in e3 if x["learner"] == name]
    R = [x["R"] for x in pts]; ref = pts[0]["V_ref"]
    ax.errorbar(R, [x["mm"] / ref for x in pts], yerr=[x["mm_sd"] / ref for x in pts], color=c, marker="o", ms=3, capsize=2, label=name)
    ax.plot(R, [x["plugin"] / ref for x in pts], color=c, ls="--", lw=1.2)
ax.axhline(1, color=MUTED, lw=0.8); ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 40, 80]); ax.set_xticklabels(["5", "10", "20", "40", "80"])
ax.set_xlabel("independent training runs R"); ax.set_ylabel("estimate / reference (R=200)"); ax.legend(fontsize=7, loc="lower right")
fig.savefig(D + "e3_est.pdf", bbox_inches="tight"); fig.savefig(D + "e3_est.png", dpi=150, bbox_inches="tight")
print("ok")
