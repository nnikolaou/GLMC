"""Figure 1 (left), refined: margin-normalised excess risk against margin-weighted expressed information, with the
two-sided band implied by Theorem 9 for binary labels: h^{-1}(Vbar) <= Exc/mbar <= Vbar/2 + modal bias/mbar.
Vbar = E[|2eta-1| V(X)] / mbar, mbar = E|2eta-1| = 1 - 1.5 eps in the running example. Uses e1_results.json only."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
BLUE, GREY, INK, MUTED = "#2a78d6", "#9a9a94", "#222222", "#6b6b66"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e3",
                     "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42})
h = lambda p: -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
hinv = lambda v: 0.0 if v <= 0 else (0.5 if v >= 1 else brentq(lambda p: h(p) - v, 1e-15, 0.5))
e1 = json.load(open("e1_results.json")); pts = []
for x in e1:
    m = 1 - 1.5 * x["eps"]; pts.append(dict(v=2 * (x["G4_upper"] - x["bias"]) / m, y=x["excess_risk"] / m, b=x["bias"] / m, interp=x["train_err"] < 0.01))
viol = sum(not (hinv(p["v"]) - 1e-9 <= p["y"] <= p["v"] / 2 + p["b"] + 1e-9) for p in pts)
fig, ax = plt.subplots(figsize=(3.4, 2.6)); v = np.linspace(0, 1, 200); lo = np.array([hinv(t) for t in v])
ax.fill_between(v, lo, v / 2, color=BLUE, alpha=0.10, lw=0, label="band of Thm. 9 (no modal bias)")
ax.plot(v, v / 2, color=MUTED, lw=1, ls="--"); ax.plot(v, lo, color=MUTED, lw=1, ls=":")
for interp, c, lab in [(False, GREY, "non-interpolating"), (True, BLUE, "interpolating")]:
    P = [p for p in pts if p["interp"] == interp and p["b"] <= 0.01]; Q = [p for p in pts if p["interp"] == interp and p["b"] > 0.01]
    ax.scatter([p["v"] for p in P], [p["y"] for p in P], s=14, color=c, label=lab, zorder=3)
    ax.scatter([p["v"] for p in Q], [p["y"] for p in Q], s=16, facecolors="none", edgecolors=c, lw=1, zorder=3,
               label="modal bias > 1%" if interp is False else None)
ax.text(0.34, 0.205, "upper: $\\bar V/2$", color=MUTED, fontsize=7, ha="right"); ax.text(0.70, 0.13, "lower: $h^{-1}(\\bar V)$", color=MUTED, fontsize=7)
ax.set_xlabel("margin-weighted expressed information $\\bar V$ (bits)"); ax.set_ylabel("excess risk / mean margin")
ax.set_xlim(0, 1); ax.set_ylim(0, 0.52); ax.legend(fontsize=6.5, loc="upper left")
fig.savefig("paper/figs/e1_scatter.pdf", bbox_inches="tight")
fig.savefig("paper/figs/e1_scatter.png", dpi=200, bbox_inches="tight")
print("violations:", viol, "of", len(pts), "; with modal bias > 1%:", sum(p["b"] > 0.01 for p in pts))
