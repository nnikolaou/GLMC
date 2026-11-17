"""Model selection WITHIN the a-priori overparameterized candidate set (configurations designed to interpolate),
on the R6 cost-matched single-dataset protocol. Criteria computed exactly as in r6_analyse.py, restricted to the set."""
import numpy as np, glob, json
from r6_analyse import est_T, safe_inv, vstats, labelfree, asym
OVER = ["kNN k=1", "Tree depth=None", "RF leaf=1", "GB 300x d6 lr1", "SVM-RBF C=10000", "MLP512 alpha=1e-05"]
rows = []
for f in sorted(glob.glob("r6_*.npz")):
    z = np.load(f); names = list(z["names"]); idx = np.array([names.index(n) for n in OVER]); K = int(z["K"])
    yte, yn = z["yte"], z["ytr_noisy"]; t = z["times"][idx].sum(0)
    err = (z["test_refit"][idx] != yte).mean(1); best = err.min(); reg = lambda c: float(err[np.argmin(c)] - best)
    r = dict(file=f, random=float(err.mean() - best), cost_holdout1=float(t[1]))
    r["reg_holdout1"] = reg((z["hold_pred"][idx] != yn[z["hold_idx"]]).mean(1))
    for Kf, ci, ui in ((2, 2, 4), (5, 3, 5)):
        oof = z[f"oof{Kf}"][idx]; r[f"reg_cv{Kf}"] = reg((oof != yn).mean(1))
        Te = est_T(z[f"oof{Kf}"], yn, K); Ti, _ = safe_inv(Te, K)        # T estimated from all 19 configurations' OOF consensus
        r[f"reg_cv{Kf}_bc"] = reg((1 - Ti[yn[None, :], oof]).mean(1))
        V, pib, modes = vstats(z[f"testfold{Kf}"][idx], K); r[f"reg_V{Kf}"] = reg(V)
        Vall, piball, modesall = vstats(z[f"testfold{Kf}"], K)             # consensus over all 19 candidates (diverse reference)
        cons_lf = labelfree(modesall, piball, K)[idx]; r[f"reg_lf{Kf}"] = reg(cons_lf)
        r[f"cost_cv{Kf}"] = float(t[ci]); r[f"cost_unlab{Kf}"] = float(t[ui])
        tall = z["times"].sum(0); r[f"cost_pool{Kf}"] = float(tall[ci] + tall[ui]); r[f"cost_poolfit{Kf}"] = float(tall[ci])   # fits + predictions of the full 19-config reference pool
    rows.append(r)
json.dump(rows, open("r6_overparam.json", "w"), indent=1)
def boot(a, b, R, B=4000):
    d = np.array([x["reg_" + a] - x["reg_" + b] for x in R]); g = np.random.default_rng(0)
    bs = [g.choice(d, len(d)).mean() for _ in range(B)]; return d.mean(), np.percentile(bs, [2.5, 97.5])
keys = ["holdout1", "cv2", "cv5", "cv5_bc", "V2", "V5", "lf2", "lf5"]
for g, lab in [("sym0.2", "symmetric"), ("classdep", "class-dependent"), ("instdep", "instance-dependent"), ("", "ALL")]:
    R = [x for x in rows if g in x["file"]]
    print(f"{lab:18s} n={len(R)} random={np.mean([x['random'] for x in R]):.4f} | " + " ".join(f"{k}={np.mean([x['reg_'+k] for x in R]):.4f}" for k in keys))
    print("   V5 - cv5 = %+.4f [%+.4f,%+.4f]" % ((lambda d, c: (d, c[0], c[1]))(*boot("V5", "cv5", R))),
          "  V5 - holdout1 = %+.4f [%+.4f,%+.4f]" % ((lambda d, c: (d, c[0], c[1]))(*boot("V5", "holdout1", R))),
          "  V2 - holdout1 = %+.4f [%+.4f,%+.4f]" % ((lambda d, c: (d, c[0], c[1]))(*boot("V2", "holdout1", R))))
R = rows
print("exact-best picks: ", {k: sum(x['reg_' + k] < 1e-12 for x in R) for k in keys}, "of", len(R))
print("median cost ratios vs holdout1: cv2 %.2f cv5 %.2f unlab5/cv5 %.3f" % (np.median([x['cost_cv2'] / x['cost_holdout1'] for x in R]),
      np.median([x['cost_cv5'] / x['cost_holdout1'] for x in R]), np.median([x['cost_unlab5'] / x['cost_cv5'] for x in R])))
