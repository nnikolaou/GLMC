import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9a94", "#222222", "#6b6b66"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
    "lines.linewidth": 1.8, "legend.frameon": False, "pdf.fonttype": 42})
D = "paper/figs/"
# R2: 3 rows (legacy, V, test error) x 4 datasets, rho=0.2 (blue) and rho=0 (grey), lr=0.1
r2 = json.load(open("r2_results.json")); dss = ["wdbc", "biopsy", "oj", "caravan"]
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.0), sharex=True)
for j, ds in enumerate(dss):
    for rho, c, lab in [(0.0, GREY, "no injected noise"), (0.2, BLUE, "20% label noise")]:
        t = [x for x in r2 if x["dataset"] == ds and x["rho"] == rho and x["lr"] == 0.1]; rr = [x["round"] for x in t]
        for i, k in enumerate(["legacy_IFX", "V", "test_err"]):
            axs[i, j].plot(rr, [x[k] for x in t], color=c, label=lab)
        b = min(t, key=lambda x: x["test_err"])
        if rho == 0.2: axs[2, j].plot([b["round"]], [b["test_err"]], "o", color=BLUE, ms=4)
    axs[0, j].set_title(ds, fontsize=9)
for i, lab in enumerate(["old plane: $I_S(F;X)/\\log n$", "expressed info V (bits)", "clean test error"]):
    axs[i, 0].set_ylabel(lab, fontsize=8)
for j in range(4): axs[2, j].set_xlabel("boosting round")
axs[0, 0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(D + "r2_boost_real.pdf", bbox_inches="tight"); fig.savefig(D + "r2_boost_real.png", dpi=140, bbox_inches="tight")
# R3: digits & wdbc, rho=0.2: test err and V vs p/n, min-norm vs ridge
r3 = json.load(open("r3_results.json"))
fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.4), sharex=True)
for j, ds in enumerate(["digits", "wdbc"]):
    for lam, c, lab in [(0.0, BLUE, "min-norm"), (0.01, GREY, "ridge λ = 0.01")]:
        t = [x for x in r3 if x["dataset"] == ds and x["rho"] == 0.2 and x["lam"] == lam]
        for i, k in enumerate(["test_err", "V"]):
            axs[i, j].plot([x["ratio"] for x in t], [x[k] for x in t], color=c, marker="o", ms=3, label=lab)
    for i in range(2): axs[i, j].axvline(1, color=MUTED, lw=0.8, ls=(0, (4, 3))); axs[i, j].set_xscale("log", base=2)
    axs[0, j].set_title(f"{ds} (K = {10 if ds=='digits' else 2}), 20% label noise", fontsize=9); axs[1, j].set_xlabel("width p / n")
axs[0, 0].set_ylabel("clean test error"); axs[1, 0].set_ylabel("expressed info V (bits)"); axs[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(D + "r3_dd_real.pdf", bbox_inches="tight"); fig.savefig(D + "r3_dd_real.png", dpi=140, bbox_inches="tight")
print("ok")
# R1: V vs clean test error, rho=0.2, per dataset
r1 = json.load(open("r1_results.json"))
fig, axs = plt.subplots(1, 5, figsize=(9.6, 2.3))
for ax, ds in zip(axs, ["wdbc", "biopsy", "oj", "caravan", "digits"]):
    t = [x for x in r1 if x["dataset"] == ds and x["rho"] == 0.2]
    I = [x for x in t if x["train_err"] < 0.02]; N = [x for x in t if x["train_err"] >= 0.02]
    ax.scatter([x["V"] for x in N], [x["test_err"] for x in N], s=12, color=GREY, label="does not interpolate")
    ax.scatter([x["V"] for x in I], [x["test_err"] for x in I], s=14, color=BLUE, label="interpolates")
    if I:
        b = min(I, key=lambda x: x["V"]); ax.scatter([b["V"]], [b["test_err"]], s=60, facecolors="none", edgecolors=INK, lw=1)
    ax.set_title(ds, fontsize=9); ax.set_xlabel("V (bits)")
axs[0].set_ylabel("clean test error"); axs[0].legend(fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(D + "r1_real_scatter.pdf", bbox_inches="tight"); fig.savefig(D + "r1_real_scatter.png", dpi=140, bbox_inches="tight")
print("ok r1")
