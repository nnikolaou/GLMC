"""R5: noise-robust selection criteria.
Backward-corrected holdout (Natarajan et al. 2013; Patrini et al. 2017): with T[y, y~] = P(noisy y~ | clean y),
the per-point loss 1 - Tinv[y~, yhat] is unbiased for the clean 0-1 loss 1{yhat != y}.
T is either the oracle matrix or estimated from the noisy TRAINING labels against the out-of-fold consensus
of all candidate configurations (no validation labels used). Also a fully label-free criterion that measures
bias as disagreement with the cross-configuration consensus on unlabeled inputs."""
import numpy as np, glob, json

KEYS = ["holdout", "holdout_clean", "holdout_bc_oracle", "holdout_bc_est", "V", "combined", "combined_bc_est",
        "labelfree", "gated", "gated_bc_est"]

def oracle_T(kind, K):
    if kind.startswith("sym"):
        r = float(kind[3:]); return (1 - r) * np.eye(K) + r / (K - 1) * (np.ones((K, K)) - np.eye(K))
    if K == 2: return np.array([[0.6, 0.4], [0.05, 0.95]])
    return 0.65 * np.eye(K) + 0.35 * np.roll(np.eye(K), 1, axis=1)

def mode_rows(P, K):  # P (..., R, M) -> counts over axis -2
    counts = np.stack([(P == k).sum(-2) for k in range(K)], -2)
    return counts.argmax(-2), counts

def safe_inv(T, K, lam=0.05):
    T = (1 - lam) * T + lam * np.eye(K)
    if np.linalg.cond(T) > 50: return np.eye(K), False
    return np.linalg.inv(T), True

def est_T(Pp, Mtr, Yn, K):
    L, R, n = Pp.shape; C = np.zeros((K, K))
    for r in range(R):
        oof = ~Mtr[r]
        cons, _ = mode_rows(Pp[:, r, oof], K)   # mode over the configurations (axis -2)
        np.add.at(C, (cons, Yn[r, oof]), 1)
    rows = C.sum(1, keepdims=True)
    T = np.where(rows > 0, C / np.maximum(rows, 1), np.eye(K))
    return T

def stats(Pv, Pt, K):
    L = Pv.shape[0]; V = np.zeros(L); pibar = np.zeros(L); mt = np.zeros((L, Pt.shape[2]), int); mv = np.zeros((L, Pv.shape[2]), int)
    for l in range(L):
        m, c = mode_rows(Pt[l], K); q = c / Pt.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(np.where(q > 0, q * np.log2(q), 0)).sum(0)
        V[l] = (H + ((c > 0).sum(0) - 1) / (2 * Pt.shape[1] * np.log(2))).mean(); pibar[l] = (1 - q.max(0)).mean(); mt[l] = m
        mv[l], _ = mode_rows(Pv[l], K)
    return V, pibar, mt, mv

def criteria(Pv, yv, yvc, K, cols, V, pibar, mt, mv, Tinv_o, Tinv_e):
    L = Pv.shape[0]; m = len(cols); c = {}
    P = Pv[:, :, cols]; y = yv[cols]
    c["holdout"] = (P != y).mean((1, 2))
    c["holdout_clean"] = (P != yvc[cols]).mean((1, 2))
    lo = 1 - Tinv_o[y[None, None, :], P]; le = 1 - Tinv_e[y[None, None, :], P]
    c["holdout_bc_oracle"] = lo.mean((1, 2)); c["holdout_bc_est"] = le.mean((1, 2))
    c["V"] = V
    em = (mv[:, cols] != y).mean(1); c["combined"] = em + pibar * np.clip(1 - em * K / (K - 1), 0, 1)
    em_bc = (1 - Tinv_e[y[None, :], mv[:, cols]]).mean(1)
    c["combined_bc_est"] = em_bc + pibar * np.clip(1 - em_bc * K / (K - 1), 0, 1)
    cons = np.array([np.bincount(col, minlength=K).argmax() for col in mt.T])
    D = (mt != cons).mean(1); c["labelfree"] = D + pibar * np.clip(1 - D * K / (K - 1), 0, 1)
    b = c["holdout"].argmin(); se = np.sqrt(c["holdout"][b] * (1 - c["holdout"][b]) / m)
    c["gated"] = np.where(c["holdout"] <= c["holdout"][b] + se, V, np.inf)
    pe = le.mean(1); b = c["holdout_bc_est"].argmin(); se = pe[b].std() / np.sqrt(m)
    c["gated_bc_est"] = np.where(c["holdout_bc_est"] <= c["holdout_bc_est"][b] + se, V, np.inf)
    return c

def analyse(m_small=40, draws=100, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for f in sorted(glob.glob("r5_*.npz")):
        z = np.load(f); ds, rest = f[3:-4].split("_", 1); kind, sd = rest.rsplit("_s", 1)
        Pv, Pt, yv, yvc, yt, K = z["Pv"], z["Pt"], z["yv"], z["yv_clean"], z["yt"], int(z["K"])
        test = (Pt != yt).mean((1, 2)); interp = z["train_err"] < 0.02
        To = oracle_T(kind, K); Te = est_T(z["Pp"], z["Mtr"], z["Yn"], K)
        Tinv_o, _ = safe_inv(To, K, 0.0); Tinv_e, ok = safe_inv(Te, K)
        V, pibar, mt, mv = stats(Pv, Pt, K)
        for mlab, colsets in [("full", [np.arange(len(yv))]),
                              (f"m={m_small}", [rng.choice(len(yv), m_small, replace=False) for _ in range(draws)])]:
            reg = {k: [] for k in KEYS}; regI = {k: [] for k in KEYS}
            for cols in colsets:
                c = criteria(Pv, yv, yvc, K, cols, V, pibar, mt, mv, Tinv_o, Tinv_e)
                for k in KEYS:
                    reg[k].append(test[np.argmin(c[k])] - test.min())
                    if interp.sum() > 3:
                        ci = np.where(interp, c[k], np.inf)
                        if np.isinf(ci).all(): ci = np.where(interp, V, np.inf)   # gate removed every interpolator
                        regI[k].append(test[np.argmin(ci)] - test[interp].min())
            row = dict(dataset=ds, noise=kind, split=int(sd), val=mlab, n_val=len(yv), K=K, T_est_ok=bool(ok),
                       T_err=float(np.abs(Te - To).max()), random=float(test.mean() - test.min()),
                       **{f"reg_{k}": float(np.mean(v)) for k, v in reg.items()})
            if interp.sum() > 3: row.update({f"regI_{k}": float(np.mean(v)) for k, v in regI.items()})
            rows.append(row)
    json.dump(rows, open("r5_analysis.json", "w"), indent=1)
    return rows

def boot(rows, a, b, ns, val, prefix="reg_", B=5000, seed=0):
    R = [r for r in rows if r["noise"] in ns and r["val"] == val and prefix + a in r]
    d = np.array([r[prefix + a] - r[prefix + b] for r in R]); g = np.random.default_rng(seed)
    bs = [g.choice(d, len(d)).mean() for _ in range(B)]
    return d.mean(), np.percentile(bs, [2.5, 97.5]), len(d)

if __name__ == "__main__":
    rows = analyse()
    for ns, lab in [(("sym0.2", "sym0.3"), "symmetric"), (("classdep",), "class-dependent")]:
        for val in ["full", "m=40"]:
            R = [r for r in rows if r["noise"] in ns and r["val"] == val]
            print(f"\n{lab} {val} n={len(R)}  T_err max-abs mean={np.mean([r['T_err'] for r in R]):.3f}")
            for k in KEYS:
                d, ci, n = boot(rows, k, "holdout", ns, val)
                RI = [r for r in R if "regI_" + k in r]
                print(f"  {k:18s} all {np.mean([r['reg_'+k] for r in R]):.4f}  vs holdout {d:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]"
                      f"   interp {np.mean([r['regI_'+k] for r in RI]):.4f}")
