"""Remedies for shared bias, on the 90 real-data selection cases (r5_*.npz):
(a) clustered consensus: one vote per cluster of near-duplicate candidates (average linkage on 1 - agreement, cut 0.05);
(b) switch rule: if the transition matrix estimated from the training labels is asymmetric, use corrected holdout,
    otherwise the label-free criterion. Asymmetry = max |T_hat - T_hat_sym| over off-diagonals, where T_hat_sym
    replaces each row's off-diagonal mass by its uniform spread at the matrix's mean noise rate."""
import numpy as np, glob, json
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from r5_analyse import stats, est_T, safe_inv, oracle_T

def plurality(S, K): return np.array([np.bincount(col, minlength=K).argmax() for col in S.T])

def cons_clustered(mt, K, tau=0.05):
    A = (mt[:, None, :] == mt[None, :, :]).mean(2); D = 1 - A; np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "average"), t=tau, criterion="distance")
    votes = np.stack([plurality(mt[lab == c], K) for c in np.unique(lab)])
    return plurality(votes, K), len(np.unique(lab))

def asym(T):
    K = len(T); rho = 1 - np.trace(T) / K
    Ts = (1 - rho) * np.eye(K) + rho / (K - 1) * (np.ones((K, K)) - np.eye(K))
    return float(np.abs(T - Ts).max())

rows = []
for f in sorted(glob.glob("r5_*.npz")):
    z = np.load(f); ds, rest = f[3:-4].split("_", 1); kind, sd = rest.rsplit("_s", 1)
    Pv, Pt, yv, yt, K = z["Pv"], z["Pt"], z["yv"], z["yt"], int(z["K"])
    test = (Pt != yt).mean((1, 2)); V, pibar, mt, mv = stats(Pv, Pt, K)
    def lf(cons):
        D = (mt != cons).mean(1); return D + pibar * np.clip(1 - D * K / (K - 1), 0, 1)
    c_plain = plurality(mt, K); c_cl, ncl = cons_clustered(mt, K)
    Te = est_T(z["Pp"], z["Mtr"], z["Yn"], K); Ti, _ = safe_inv(Te, K)
    hold_bc = (1 - Ti[yv[None, None, :], Pv]).mean((1, 2))
    a = asym(Te)
    r = dict(dataset=ds, noise=kind, split=int(sd), asym=a, nclusters=ncl,
             labelfree=float(test[lf(c_plain).argmin()] - test.min()),
             labelfree_clustered=float(test[lf(c_cl).argmin()] - test.min()),
             holdout=float(test[(Pv != yv).mean((1, 2)).argmin()] - test.min()),
             holdout_bc=float(test[hold_bc.argmin()] - test.min()))
    rows.append(r)
json.dump(rows, open("r5_remedy.json", "w"), indent=1)
for g, lab in [(("sym0.2", "sym0.3"), "symmetric"), (("classdep",), "class-dependent")]:
    R = [r for r in rows if r["noise"] in g]
    print(f"{lab:16s} n={len(R)} asym mean={np.mean([r['asym'] for r in R]):.3f} range=[{min(r['asym'] for r in R):.3f},{max(r['asym'] for r in R):.3f}] "
          + " ".join(f"{k}={np.mean([r[k] for r in R]):.4f}" for k in ["holdout", "holdout_bc", "labelfree", "labelfree_clustered"])
          + f" nclusters={np.mean([r['nclusters'] for r in R]):.1f}")
for thr in [0.05, 0.1, 0.15, 0.2]:
    sw = [r["holdout_bc"] if r["asym"] > thr else r["labelfree"] for r in rows]
    sw_s = np.mean([s for s, r in zip(sw, rows) if r["noise"] != "classdep"]); sw_c = np.mean([s for s, r in zip(sw, rows) if r["noise"] == "classdep"])
    print(f"switch rule thr={thr}: symmetric {sw_s:.4f} class-dep {sw_c:.4f} overall {np.mean(sw):.4f}  (switched {sum(r['asym']>thr for r in rows)}/90)")
print("reference overall: holdout %.4f holdout_bc %.4f labelfree %.4f" % tuple(np.mean([r[k] for r in rows]) for k in ["holdout", "holdout_bc", "labelfree"]))
