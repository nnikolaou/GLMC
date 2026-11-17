"""Analysis of the consensus stress tests (s_consensus.py output)."""
import numpy as np, glob, json
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def consensus(M, members, cluster_tau=None):
    """M: (J, n) modal predictions (binary). Majority vote over members, optionally one vote per cluster."""
    S = M[members]
    if cluster_tau is None or len(members) < 2:
        return (S.mean(0) > 0.5).astype(int), len(members)
    A = (S[:, None, :] == S[None, :, :]).mean(2); Dm = 1 - A; np.fill_diagonal(Dm, 0)
    lab = fcluster(linkage(squareform(Dm, checks=False), "average"), t=cluster_tau, criterion="distance")
    votes = np.stack([(S[lab == c].mean(0) > 0.5) for c in np.unique(lab)])
    return (votes.mean(0) > 0.5).astype(int), len(np.unique(lab))

def load_case(f):
    z = np.load(f); Pt, Pv, yv, e = z["Pt"], z["Pv"], z["yv"], z["eta_t"]; fr = float(z["f"])
    b = (e > 0.5).astype(int); w = np.abs(2 * e - 1)
    exc = (w * (Pt != b)).mean((1, 2))
    p1 = Pt.mean(1); mode = (p1 >= 0.5).astype(int); pi = np.minimum(p1, 1 - p1)
    R = Pt.shape[1]; V = (np.where(pi > 0, h2(pi), 0) + (pi > 0) / (2 * R * np.log(2))).mean(1)
    beta = (mode != b).mean(1)                       # unweighted modal-bias mass
    hold = (Pv != yv).mean(2)                         # (J, R) per-run noisy validation errors
    T = np.array([[1 - fr, fr], [0.0, 1.0]]); Ti = np.linalg.inv(T)
    hold_bc_pt = 1 - Ti[yv[None, None, :], Pv]        # (J, R, nv) corrected per-point losses
    return dict(exc=exc, mode=mode, pibar=pi.mean(1), V=V, beta=beta, b=b, e=e, fam=z["fam"], Pv=Pv, yv=yv,
                hold_bc_pt=hold_bc_pt, f=fr)

def select_all(c, pool, rng, m_small=40, draws=30, taus=(None, 0.05)):
    out = {}; exc = c["exc"][pool]; best = exc.min(); M = c["mode"]
    for tau in taus:
        cons, ncl = consensus(M, pool, tau)
        D = (M[pool] != cons).mean(1); lf = D + c["pibar"][pool] * np.clip(1 - 2 * D, 0, 1)
        key = "labelfree" + ("" if tau is None else "_clustered")
        out[key] = exc[lf.argmin()] - best
        out["delta" + ("" if tau is None else "_clustered")] = float((cons != c["b"]).mean())
        out["nclusters" + ("" if tau is None else "_clustered")] = ncl
        # theory check: bias-only selection by D is within 2*delta of the best modal-bias mass
        bsel = c["beta"][pool][D.argmin()]
        out["bias_gap" + ("" if tau is None else "_clustered")] = float(bsel - c["beta"][pool].min())
    # adaptive: one vote per cluster only if some cluster holds >= 1/3 of the candidates
    S = M[pool]; A = (S[:, None, :] == S[None, :, :]).mean(2); Dm = 1 - A; np.fill_diagonal(Dm, 0)
    lab = fcluster(linkage(squareform(Dm, checks=False), "average"), t=0.05, criterion="distance")
    use = np.bincount(lab).max() >= len(pool) / 3
    cons = consensus(M, pool, 0.05 if use else None)[0]
    D = (S != cons).mean(1); lf = D + c["pibar"][pool] * np.clip(1 - 2 * D, 0, 1)
    out["labelfree_adaptive"] = exc[lf.argmin()] - best; out["adaptive_used"] = bool(use)
    out["V"] = exc[c["V"][pool].argmin()] - best
    Pv, yv = c["Pv"][pool], c["yv"]
    out["holdout"] = exc[(Pv != yv).mean((1, 2)).argmin()] - best
    out["holdout_bc"] = exc[c["hold_bc_pt"][pool].mean((1, 2)).argmin()] - best
    sm, smb = [], []
    for _ in range(draws):
        cols = rng.choice(len(yv), m_small, replace=False)
        sm.append(exc[(Pv[:, :, cols] != yv[cols]).mean((1, 2)).argmin()] - best)
        smb.append(exc[c["hold_bc_pt"][pool][:, :, cols].mean((1, 2)).argmin()] - best)
    out["holdout40"] = float(np.mean(sm)); out["holdout_bc40"] = float(np.mean(smb))
    return out

def run(pool_size=12, shares=(0, 3, 6, 8, 9), npools=20, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for f in sorted(glob.glob("sc_f*_s*.npz")):
        c = load_case(f); lin = np.flatnonzero(c["fam"] == 0); div = np.flatnonzero(c["fam"] == 1)
        e = c["e"]; fr = c["f"]; thr = (0.5 - fr) / (1 - fr)
        delta_noise = float(((e > thr) & (e <= 0.5)).mean())   # mass where the noisy Bayes rule differs from b*
        for k in shares:
            if fr > 0 and k not in (0,): continue              # noise sweep uses the diverse-only pool
            for _ in range(npools if k not in (0,) or fr == 0 else 1):
                pool = np.concatenate([rng.choice(lin, k, replace=False), rng.choice(div, min(pool_size - k, len(div)), replace=False)]) if k else div
                r = select_all(c, pool, rng); r.update(file=f, f=fr, linear_share=k / len(pool), pool=len(pool), delta_noise=delta_noise)
                rows.append(r)
    json.dump(rows, open("sc_analysis.json", "w"), indent=1)
    return rows

if __name__ == "__main__":
    rows = run()
    keys = ["holdout", "holdout40", "holdout_bc", "holdout_bc40", "V", "labelfree", "labelfree_clustered", "labelfree_adaptive"]
    print("S1: linear share sweep (f = 0)")
    for s in sorted({r["linear_share"] for r in rows if r["f"] == 0}):
        R = [r for r in rows if r["f"] == 0 and r["linear_share"] == s]
        print(f" share {s:.2f} n={len(R):3d} delta={np.mean([r['delta'] for r in R]):.3f} delta_cl={np.mean([r['delta_clustered'] for r in R]):.3f} "
              f"ncl={np.mean([r['nclusters_clustered'] for r in R]):.1f} | " + " ".join(f"{k}={np.mean([r[k] for r in R]):.4f}" for k in keys))
    print("S2: class-dependent noise sweep (diverse pool)")
    for fr in sorted({r["f"] for r in rows}):
        R = [r for r in rows if r["f"] == fr and r["linear_share"] == 0]
        print(f" f={fr:.1f} n={len(R)} delta={np.mean([r['delta'] for r in R]):.3f} delta_noise={np.mean([r['delta_noise'] for r in R]):.3f} | "
              + " ".join(f"{k}={np.mean([r[k] for r in R]):.4f}" for k in keys))
    viol = [r for r in rows if r["bias_gap"] > 2 * r["delta"] + 1e-12]
    print("bound check: bias gap <= 2 delta violated in", len(viol), "of", len(rows))
