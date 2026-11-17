"""Analysis of R9 (regression, Protocol B analogue). Regret = clean test MSE of the chosen configuration's full refit minus the best's.
Criteria: holdout, 2-fold CV, 5-fold CV, 2-fold CV over 3 splits (held-out labels); min-Var2 (two disjoint halves; = min-V at low
resolution, main Thm 13), min-Var2 averaged over 3 disjoint pairs, and the regression consensus criterion (no held-out labels)."""
import numpy as np, glob, json, re, os
from collections import defaultdict
from r9_regression import OVER

def consensus(MH, V):
    """MH: (L, M) mean of the two half-models per candidate; V: (L, M) pointwise half-sample variance estimates.
    Score_j = mean[(m_j - c)^2 + v_j/2], c = pointwise median of the candidates' means (all candidates of the pool)."""
    c = np.median(MH, 0); return ((MH - c) ** 2 + V / 2).mean(1)

def analyse_case(f):
    z = dict(np.load(f)); fb = f.replace("r9_", "r9b_")
    if os.path.exists(fb):                                   # merge the second batch of candidates (same data and splits)
        zb = np.load(fb); assert (zb["ytr_noisy"] == z["ytr_noisy"]).all() and (zb["fold2"] == z["fold2"]).all()
        for k in ("names", "test_refit", "hold_pred", "oof2", "oof5", "testfold2", "testfold5", "extra_test", "extra_oof", "times"):
            z[k] = np.concatenate([z[k], zb[k]], 0)
    names = list(z["names"]); L = len(names); yte, yn = z["yte"], z["ytr_noisy"]; t = z["times"]
    err = ((z["test_refit"] - yte) ** 2).mean(1)
    hold = ((z["hold_pred"] - yn[z["hold_idx"]]) ** 2).mean(1)
    cv2 = ((z["oof2"] - yn) ** 2).mean(1); cv5 = ((z["oof5"] - yn) ** 2).mean(1)
    cv2x3 = (cv2 + ((z["extra_oof"] - yn) ** 2).mean(2).sum(1)) / 3
    pairs = [z["testfold2"]] + [z["extra_test"][:, d] for d in range(z["extra_test"].shape[1])]     # each (L, 2, M)
    Vpt = [(p[:, 0] - p[:, 1]) ** 2 / 2 for p in pairs]; MH = [(p[:, 0] + p[:, 1]) / 2 for p in pairs]
    V2 = Vpt[0].mean(1); V2x3 = np.mean([v.mean(1) for v in Vpt], 0)
    crit = dict(holdout=hold, cv2=cv2, cv5=cv5, cv2x3=cv2x3, V2=V2, V2x3=V2x3, train_free_random=None)
    out = dict(file=f)
    pools = (("all", np.arange(L)), ("op", np.array([names.index(n) for n in OVER if n in names])),
             ("op7", np.array([names.index(n) for n in OVER[:7]])))
    for pool, idx in pools:
        e = err[idx]; best = e.min(); r = {}
        for k, c in crit.items():
            if c is None: continue
            r[k] = float(e[np.argmin(c[idx])] - best)
        r["lf2"] = float(e[np.argmin(consensus(MH[0], Vpt[0])[idx])] - best)            # consensus over all 17 candidates
        r["lf2x3"] = float(e[np.argmin(np.mean([consensus(m, v) for m, v in zip(MH, Vpt)], 0)[idx])] - best)
        r["random"] = float(np.median(e) - best); r["best_name"] = names[idx[np.argmin(e)]]; r["best_err"] = float(best)
        # "random" = regret of the median candidate (the mean is dominated by catastrophic interpolators)
        T = t.sum(0); Ti = t[idx].sum(0); h1 = Ti[1]          # cols: refit, holdout, cv2, cv5, unlab2, unlab5, extra, unlab_extra
        r["cost"] = dict(holdout=1.0, cv2=Ti[2] / h1, cv5=Ti[3] / h1, cv2x3=(Ti[2] + Ti[6]) / h1, V2=(Ti[2] + Ti[4]) / h1,
                         V2x3=(Ti[2] + Ti[4] + Ti[6] + Ti[7]) / h1, lf2=(T[2] + T[4]) / h1, lf2x3=(T[2] + T[4] + T[6] + T[7]) / h1)
        r["exact"] = {k: r[k] < 1e-12 for k in list(crit)[:-1] + ["lf2", "lf2x3"]}
        out[pool] = r
    return out

def key(f):
    m = re.match(r"r9_(.+)_(g[0-9.]+)_s(\d+)\.npz", f); return m.group(1), m.group(2), int(m.group(3))

def cboot(rows, pool, a, b, B=4000, seed=0):
    C = defaultdict(list)
    for r in rows: ds, nz, s = key(r["file"]); C[(ds, nz)].append(r[pool][a] - r[pool][b])
    C = [np.array(v) for v in C.values()]; g = np.random.default_rng(seed); pt = np.concatenate(C).mean()
    bs = [np.concatenate([C[i] for i in g.integers(0, len(C), len(C))]).mean() for _ in range(B)]
    return float(pt), [float(x) for x in np.percentile(bs, [2.5, 97.5])]

KEYS = ["holdout", "cv2", "cv5", "cv2x3", "V2", "V2x3", "lf2", "lf2x3"]
if __name__ == "__main__":
    rows = [analyse_case(f) for f in sorted(glob.glob("r9_*_s[0-9].npz"))]
    print(len(rows), "cases")
    G = [("g0.5", "sigma=0.5"), ("g1.0", "sigma=1.0"), ("", "all")]; summ = {}
    for pool in ("op7", "op", "all"):
        summ[pool] = {k: {lab: float(np.mean([r[pool][k] for r in rows if g in r["file"]])) for g, lab in G} for k in KEYS + ["random"]}
        summ[pool]["_cost"] = {k: float(np.median([r[pool]["cost"][k] for r in rows])) for k in rows[0][pool]["cost"]}
        summ[pool]["_exact"] = {k: int(sum(r[pool]["exact"][k] for r in rows)) for k in KEYS}
        print(f"\n=== pool {pool}   (cols: " + ", ".join(l for _, l in G) + ")")
        for k in KEYS + ["random"]:
            print(f"  {k:8s} " + " ".join(f"{summ[pool][k][l]:.4f}" for _, l in G) +
                  (f"   cost {summ[pool]['_cost'].get(k, float('nan')):.2f}  exact {summ[pool]['_exact'].get(k, '')}" if k in KEYS else ""))
        from collections import Counter
        print("  best configs:", Counter(r[pool]["best_name"] for r in rows).most_common(6))
    diffs = {}
    for pool, prs in (("op7", [("V2", "holdout"), ("V2", "cv2"), ("V2", "cv5"), ("V2x3", "cv2x3"), ("lf2", "V2")]), ("op", [("V2", "holdout"), ("V2", "cv2"), ("V2", "cv5"), ("V2x3", "cv5"), ("V2x3", "cv2x3"), ("lf2", "cv2"), ("lf2", "V2")]),
                      ("all", [("V2", "cv5"), ("lf2", "cv2"), ("lf2", "cv5"), ("lf2", "holdout"), ("lf2x3", "cv5"), ("lf2x3", "cv2x3")])):
        for a, b in prs:
            diffs[f"{pool}:{a}-{b}"] = cboot(rows, pool, a, b); print(f"  {pool}: {a}-{b} = {diffs[f'{pool}:{a}-{b}']}")
    json.dump(dict(rows=rows, summary=summ, diffs=diffs), open("r9_analysis.json", "w"), indent=1)
