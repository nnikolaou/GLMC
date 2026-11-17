"""R2: boosting trajectories on real data (old information plane vs expressed information, clean test error).
R3: random-features double descent on real data. Label noise injected in training labels only."""
import numpy as np, json, sys, warnings
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from realdata import load
from r1_real import flip, entropy_rows
warnings.filterwarnings("ignore")

def Hd(a):
    _, c = np.unique(a, return_counts=True, axis=0); p = c / c.sum(); return float(-(p * np.log2(p)).sum())

def split(ds, seed=0, test_frac=0.3):
    X, y = load(ds, seed); K = int(y.max()) + 1
    idx = np.random.default_rng(seed).permutation(len(y)); nt = int(test_frac * len(y))
    return X[idx[nt:]], y[idx[nt:]], X[idx[:nt]], y[idx[:nt]], K

def boost_run(Xp, yp, Xt, K, rho, lr, depth, T, every, r, seed):
    rng = np.random.default_rng(seed * 1000 + r)
    idx = rng.permutation(len(yp))[: len(yp) // 2]; yn = flip(yp, rho, K, rng)
    m = GradientBoostingClassifier(loss="exponential", learning_rate=lr, max_depth=depth, n_estimators=T,
                                   random_state=int(rng.integers(1e9))).fit(Xp[idx], yn[idx])
    te, legx, legy, tr = [], [], [], []
    for s, (pt, ptr) in enumerate(zip(m.staged_predict_proba(Xt), m.staged_predict_proba(Xp[idx]))):
        if (s + 1) % every: continue
        te.append((pt[:, 1] >= 0.5).astype(int))
        fb = np.minimum((ptr[:, 1] * 100).astype(int), 99); y_ = yn[idx]
        legx.append(Hd(fb) / np.log2(len(idx))); legy.append((Hd(fb) + Hd(y_) - Hd(np.stack([fb, y_], 1))) / Hd(y_))
        tr.append(np.mean((ptr[:, 1] >= 0.5) != y_))
    return np.array(te), legx, legy, tr

def r2(datasets=("wdbc", "biopsy", "oj", "caravan"), R=12, T=300, every=10, seed=0):
    out = []
    for ds in datasets:
        Xp, yp, Xt, yt, K = split(ds, seed)
        for rho in (0.0, 0.2):
            for lr, depth in ((1.0, 6), (0.1, 3)):
                res = Parallel(n_jobs=2)(delayed(boost_run)(Xp, yp, Xt, K, rho, lr, depth, T, every, r, seed + 11) for r in range(R))
                P = np.stack([x[0] for x in res], 0)  # R x steps x M
                for j in range(P.shape[1]):
                    H, Hmm, mode = entropy_rows(P[:, j], K)
                    out.append(dict(dataset=ds, rho=rho, lr=lr, depth=depth, round=(j + 1) * every,
                                    test_err=float(np.mean(P[:, j] != yt)), modal_err=float(np.mean(mode != yt)),
                                    V=float(Hmm.mean()), train_err=float(np.mean([x[3][j] for x in res])),
                                    legacy_IFX=float(np.mean([x[1][j] for x in res])),
                                    legacy_IFY=float(np.mean([x[2][j] for x in res]))))
                best = min([o for o in out if o["dataset"] == ds and o["rho"] == rho and o["lr"] == lr], key=lambda o: o["test_err"])
                print("R2", ds, rho, lr, "best round", best["round"], round(best["test_err"], 3), "final", round(out[-1]["test_err"], 3), flush=True)
                json.dump(out, open("r2_results.json", "w"), indent=1)

def rf_run(Xp, yp, Xt, K, rho, p, lam, r, seed):
    rng = np.random.default_rng(seed * 1000 + r)
    idx = rng.permutation(len(yp))[: len(yp) // 2]; yn = flip(yp, rho, K, rng)
    d = Xp.shape[1]; W = rng.normal(size=(d, p)) / np.sqrt(d); b = rng.normal(size=p) * 0.5
    F = lambda X: np.maximum(X.astype(np.float64) @ W + b, 0) / np.sqrt(p)
    Phi = F(Xp[idx]); Ytr = np.eye(K)[yn[idx]] * 2 - 1; n = len(idx)
    if lam == 0: theta = np.linalg.lstsq(Phi, Ytr, rcond=None)[0]
    elif p >= n: theta = Phi.T @ np.linalg.solve(Phi @ Phi.T + lam * n * np.eye(n), Ytr)
    else: theta = np.linalg.solve(Phi.T @ Phi + lam * n * np.eye(p), Phi.T @ Ytr)
    return (F(Xt) @ theta).argmax(1), float(np.mean((Phi @ theta).argmax(1) != yn[idx]))

def r3(datasets=("digits", "wdbc"), ratios=(0.0625, 0.125, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2, 4, 8, 16),
       R=12, seed=0):
    out = []
    for ds in datasets:
        Xp, yp, Xt, yt, K = split(ds, seed); n = len(yp) // 2
        for rho in (0.0, 0.2):
            for lam in (0.0, 1e-2):
                for ra in ratios:
                    p = max(4, int(round(ra * n)))
                    res = Parallel(n_jobs=2)(delayed(rf_run)(Xp, yp, Xt, K, rho, p, lam, r, seed + 13) for r in range(R))
                    P = np.stack([x[0] for x in res]); H, Hmm, mode = entropy_rows(P, K)
                    out.append(dict(dataset=ds, rho=rho, lam=lam, n=n, p=p, ratio=p / n, test_err=float(np.mean(P != yt)),
                                    modal_err=float(np.mean(mode != yt)), V=float(Hmm.mean()),
                                    train_err=float(np.mean([x[1] for x in res]))))
                print("R3", ds, rho, lam, flush=True)
                json.dump(out, open("r3_results.json", "w"), indent=1)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("r3", "both"): r3()
    if which in ("r2", "both"): r2()
