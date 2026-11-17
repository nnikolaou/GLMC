"""Supp. Tables S8-S9 and Figure S5 (regression extension) from r9_analysis.json, r9_exact.json and r9_exact_b.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("r9_analysis.json"))["summary"]
lab = {"holdout": "Holdout (80/20)", "cv2": "2-fold CV", "cv5": "5-fold CV", "cv2x3": "2-fold CV, 3 random splits",
       "V2": "Min-variance\\prop{} (2 disjoint halves)", "V2x3": "Min-variance\\prop{}, averaged over 3 pairs",
       "lf2": "Consensus criterion\\prop{}, 2 halves", "lf2x3": "Consensus criterion\\prop{}, 3 pairs"}
lbl = {"holdout": "yes", "cv2": "yes", "cv5": "yes", "cv2x3": "yes", "V2": "no", "V2x3": "no", "lf2": "no", "lf2x3": "no"}
cols = [("op7", "sigma=0.5"), ("op7", "sigma=1.0"), ("op7", "all"), ("op", "all"), ("all", "all")]
best = {c: min(S[c[0]][k][c[1]] for k in lab) for c in cols}; bestE = max(S["op7"]["_exact"][k] for k in lab); out = []
for k in lab:
    v = [(f"\\best{{{S[a][k][g]:.4f}}}" if abs(S[a][k][g] - best[(a, g)]) < 1e-12 else f"{S[a][k][g]:.4f}") for a, g in cols]
    e = S["op7"]["_exact"][k]; es = f"\\best{{{e}}}" if e == bestE else str(e)
    out.append(f"{lab[k]} & {lbl[k]} & " + " & ".join(v) + f" & {es} & {S['op7']['_cost'][k]:.2f}\\\\")
out.append("\\midrule\n\\emph{Median candidate} & & " + " & ".join(f"\\emph{{{S[a]['random'][g]:.3f}}}" for a, g in cols) + " & & \\\\")
open("r9_table_main.tex", "w").write("\n".join(out))
R = json.load(open("r9_exact.json")) + json.load(open("r9_exact_b.json")); out = []
for r in sorted([r for r in R if r["sigma"] == 0.5 and r["over"]], key=lambda r: r["excess"]):
    r1 = [x for x in R if x["sigma"] == 1.0 and x["learner"] == r["learner"]][0]; D = r["train_mse"]
    bd = "$\\infty$" if D < 1e-6 else f"{max(0, 0.5 * np.log2(0.25 / D)):.2f}"
    out.append(f"{r['learner']} & {D:.3f} & {bd} & {r['excess']:.3f} & {r['bias2']:.3f} & {r['var']:.3f} & {r['V2']:.3f} & {r1['excess']:.3f} & {r1['bias2']:.3f} & {r1['var']:.3f}\\\\")
open("r9_table_exact.tex", "w").write("\n".join(out))
BLUE, INK, MUTED = "#2a78d6", "#222222", "#6b6b66"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e3", "legend.frameon": False, "pdf.fonttype": 42})
fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
for ax, s in zip(axs, (0.5, 1.0)):
    P = [r for r in R if r["sigma"] == s and r["over"] and r["excess"] < 3]
    x = np.array([r["var"] for r in P]); y = np.array([r["excess"] for r in P]); b = np.array([r["bias2"] for r in P])
    sc = ax.scatter(x, y, c=b, cmap="viridis", s=28, zorder=3, vmin=0, vmax=0.32)
    t = np.linspace(0, max(x) * 1.05, 10); ax.plot(t, t, color=MUTED, lw=1, ls="--")
    for r in P:
        if r["learner"] in ("kNN k=25 dist", "kNN k=1"):
            ax.annotate(r["learner"], (r["var"], r["excess"]), (4, 3), textcoords="offset points", fontsize=6.5, color=INK)
    ax.set_xlabel("variance $\\mathbb{E}\\,v(X)$"); ax.set_ylabel("excess risk"); ax.set_title(f"$\\sigma_\\varepsilon={s}$", fontsize=8.5, loc="left")
cb = fig.colorbar(sc, ax=axs, shrink=0.85); cb.set_label("squared bias")
fig.savefig("paper/figs/r9_exact.pdf", bbox_inches="tight")
print("wrote r9_table_main.tex, r9_table_exact.tex, paper/figs/r9_exact.pdf")
