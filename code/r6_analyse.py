"""R6 analysis: regret of each selection criterion vs its compute cost, with thresholds FROZEN at the values
chosen on earlier experiments (clustering tau 0.05, trigger 1/3, asymmetry switch 0.1).
Regret = clean test error of the chosen configuration's refit model minus the best refit's.
Costs (CPU seconds, summed over the 19 configurations; the final refit is common to all criteria and excluded):
  holdout-1 = 1 fit on 80% + predict; CV-K = K fits + OOF predictions;
  V / label-free from the CV-K fold models = CV-K cost + predictions on the unlabelled pool (marginal cost = the latter)."""
import numpy as np, glob, json
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def plurality(S, K): return np.array([np.bincount(c, minlength=K).argmax() for c in S.T])

def est_T(oof, y, K):
    cons = plurality(oof, K); C = np.zeros((K, K)); np.add.at(C, (cons, y), 1)
    r = C.sum(1, keepdims=True); return np.where(r > 0, C / np.maximum(r, 1), np.eye(K))

def safe_inv(T, K, lam=0.05):
    T = (1 - lam) * T + lam * np.eye(K)
    return (np.linalg.inv(T), True) if np.linalg.cond(T) <= 50 else (np.eye(K), False)

def asym(T):
    K = len(T); rho = 1 - np.trace(T) / K
    return float(np.abs(T - ((1 - rho) * np.eye(K) + rho / (K - 1) * (np.ones((K, K)) - np.eye(K)))).max())

def vstats(TF, K):
    """TF: (L, R, M) fold-model predictions on unlabelled inputs -> V (Miller-Madow, bits), pibar, modes."""
    L, R, M = TF.shape; V = np.zeros(L); pib = np.zeros(L); modes = np.zeros((L, M), int)
    for l in range(L):
        c = np.stack([(TF[l] == k).sum(0) for k in range(K)]); q = c / R
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(np.where(q > 0, q * np.log2(q), 0)).sum(0)
        V[l] = (H + ((c > 0).sum(0) - 1) / (2 * R * np.log(2))).mean(); pib[l] = (1 - q.max(0)).mean(); modes[l] = c.argmax(0)
    return V, pib, modes

def labelfree(modes, pib, K, adaptive=False):
    cons = plurality(modes, K)
    if adaptive:
        A = (modes[:, None, :] == modes[None, :, :]).mean(2); D = 1 - A; np.fill_diagonal(D, 0)
        lab = fcluster(linkage(squareform(D, checks=False), "average"), t=0.05, criterion="distance")
        if np.bincount(lab).max() >= len(modes) / 3:
            cons = plurality(np.stack([plurality(modes[lab == c], K) for c in np.unique(lab)]), K)
    Dg = (modes != cons).mean(1); return Dg + pib * np.clip(1 - Dg * K / (K - 1), 0, 1)

def analyse_case(f):
    z = np.load(f); K = int(z["K"]); yte = z["yte"]; yn = z["ytr_noisy"]; t = z["times"]
    err = (z["test_refit"] != yte).mean(1); best = err.min(); reg = lambda c: float(err[np.argmin(c)] - best)
    out = dict(file=f, K=K, noise_rate=float((yn != z["ytr_clean"]).mean()), oracle_best=float(best),
               random=float(err.mean() - best))
    tsum = t.sum(0)   # columns: refit, holdout, cv2, cv5, unlab2, unlab5
    hold = (z["hold_pred"] != yn[z["hold_idx"]]).mean(1)
    out["reg_holdout1"] = reg(hold); out["cost_holdout1"] = float(tsum[1])
    for Kf, ci, ui in ((2, 2, 4), (5, 3, 5)):
        oof = z[f"oof{Kf}"]; cv = (oof != yn).mean(1)
        Te = est_T(oof, yn, K); Ti, _ = safe_inv(Te, K); cvbc = (1 - Ti[yn[None, :], oof]).mean(1)
        V, pib, modes = vstats(z[f"testfold{Kf}"], K)
        lf = labelfree(modes, pib, K); lfa = labelfree(modes, pib, K, adaptive=True)
        a = asym(Te); sw = cvbc if a > 0.1 else lf
        out.update({f"reg_cv{Kf}": reg(cv), f"reg_cv{Kf}_bc": reg(cvbc), f"reg_V{Kf}": reg(V), f"reg_lf{Kf}": reg(lf),
                    f"reg_lfa{Kf}": reg(lfa), f"reg_switch{Kf}": reg(sw), f"asym{Kf}": a,
                    f"cost_cv{Kf}": float(tsum[ci]), f"cost_unlab{Kf}": float(tsum[ui])})
    out["cost_refit"] = float(tsum[0])
    return out

CRIT = [("holdout1", "Holdout, 1 fit", "holdout1"), ("cv2", "2-fold CV", "cv2"), ("cv5", "5-fold CV", "cv5"),
        ("cv5_bc", "5-fold CV, noise-corrected", "cv5"), ("V5", "V from the 5 fold models", "cv5+unlab5"),
        ("lf2", "Label-free from 2 fold models", "cv2+unlab2"), ("lf5", "Label-free from 5 fold models", "cv5+unlab5"),
        ("lfa5", "Label-free, adaptive clustering (5)", "cv5+unlab5"), ("switch5", "Switch rule (5)", "cv5+unlab5")]

def cost(r, spec):
    return sum(r["cost_" + p] for p in spec.split("+"))

if __name__ == "__main__":
    rows = [analyse_case(f) for f in sorted(glob.glob("r6_*.npz"))]
    json.dump(rows, open("r6_analysis.json", "w"), indent=1)
    print(len(rows), "cases")
    for g, lab in [("sym0.2", "symmetric"), ("classdep", "class-dependent"), ("instdep", "instance-dependent")]:
        R = [r for r in rows if g in r["file"]]
        if not R: continue
        print(f"\n{lab} (n={len(R)}; mean noise rate {np.mean([r['noise_rate'] for r in R]):.2f}; random {np.mean([r['random'] for r in R]):.4f})")
        for k, name, spec in CRIT:
            rel = np.median([cost(r, spec) / r["cost_holdout1"] for r in R])
            print(f"  {name:38s} regret {np.mean([r['reg_'+k] for r in R]):.4f}   median cost x holdout-1: {rel:5.1f}")
