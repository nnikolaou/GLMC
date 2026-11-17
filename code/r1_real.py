"""R1: real data with injected symmetric label noise (training labels only; test labels clean).
For each dataset x noise x learner config: R runs, each on a random half of the pool with fresh label noise
and a fresh seed. Measures: mean clean test error, error of the modal (majority-vote) predictor, expressed
information V (entropy of the R predictions per test point, Miller-Madow corrected), train error (noisy labels),
noisy out-of-bag validation error, and the legacy empirical LMC score I_S(F;Y)/H_S(Y)."""
import numpy as np, json, sys, time, warnings
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from realdata import load, DATASETS
warnings.filterwarnings("ignore")

def configs():
    C = {}
    for k in [1, 3, 9, 27]: C[f"kNN k={k}"] = lambda k=k: KNeighborsClassifier(k)
    for d in [2, 4, 8, None]: C[f"Tree depth={d}"] = lambda d=d: DecisionTreeClassifier(max_depth=d)
    C["RF leaf=1"] = lambda: RandomForestClassifier(200, min_samples_leaf=1, n_jobs=1)
    C["RF leaf=5"] = lambda: RandomForestClassifier(200, min_samples_leaf=5, n_jobs=1)
    C["GB 100x d3 lr.1"] = lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    C["GB 300x d6 lr1"] = lambda: GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=1.0)
    for c in [0.01, 1, 1e4]: C[f"Logistic C={c:g}"] = lambda c=c: LogisticRegression(C=c, max_iter=3000)
    for c in [1, 1e4]: C[f"SVM-RBF C={c:g}"] = lambda c=c: SVC(C=c)
    for a in [1e-5, 1.0]: C[f"MLP512 alpha={a:g}"] = lambda a=a: MLPClassifier((512,), alpha=a, max_iter=500, tol=1e-6, n_iter_no_change=50)
    return C

def entropy_rows(P, K):
    """P: (R, M) int predictions -> per-column entropy (bits), plug-in and Miller-Madow, and the mode."""
    R, M = P.shape
    counts = np.stack([(P == k).sum(0) for k in range(K)], 0)   # K x M
    q = counts / R
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(np.where(q > 0, q * np.log2(q), 0)).sum(0)
    k_obs = (counts > 0).sum(0)
    Hmm = H + (k_obs - 1) / (2 * R * np.log(2))
    return H, Hmm, counts.argmax(0)

def legacy(f, y):
    def Hd(a):
        _, c = np.unique(a, return_counts=True, axis=0); p = c / c.sum(); return -(p * np.log2(p)).sum()
    Hy = Hd(y); return (Hd(f) + Hy - Hd(np.stack([f, y], 1))) / Hy

def flip(y, rho, K, rng):
    y = y.copy(); m = rng.random(len(y)) < rho
    y[m] = (y[m] + rng.integers(1, K, m.sum())) % K       # uniformly to a different class
    return y

def one(name, mk, Xp, yp, Xt, yt, K, rho, R, seed):
    P = np.zeros((R, len(yt)), int); te, tr, val, leg = [], [], [], []
    for r in range(R):
        rng = np.random.default_rng(seed * 1000 + r)
        idx = rng.permutation(len(yp)); a, b = idx[: len(yp) // 2], idx[len(yp) // 2:]
        ynoisy = flip(yp, rho, K, rng)
        m = mk()
        if hasattr(m, "random_state"): m.set_params(random_state=int(rng.integers(1e9)))
        m.fit(Xp[a], ynoisy[a])
        P[r] = m.predict(Xt); ptr = m.predict(Xp[a])
        te.append(np.mean(P[r] != yt)); tr.append(np.mean(ptr != ynoisy[a]))
        val.append(np.mean(m.predict(Xp[b]) != ynoisy[b])); leg.append(legacy(ptr, ynoisy[a]))
    H, Hmm, mode = entropy_rows(P, K)
    return dict(learner=name, test_err=float(np.mean(te)), test_err_sd=float(np.std(te)),
                modal_err=float(np.mean(mode != yt)), V_plugin=float(H.mean()), V=float(Hmm.mean()),
                train_err=float(np.mean(tr)), noisy_val_err=float(np.mean(val)), legacy=float(np.mean(leg)))

def run(R=10, rhos=(0.0, 0.1, 0.2, 0.3), test_frac=0.3, seed=0):
    out = []
    for ds in DATASETS:
        X, y = load(ds, seed); K = int(y.max()) + 1
        rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); nt = int(test_frac * len(y))
        Xt, yt, Xp, yp = X[idx[:nt]], y[idx[:nt]], X[idx[nt:]], y[idx[nt:]]
        for rho in rhos:
            t0 = time.time()
            res = Parallel(n_jobs=2)(delayed(one)(n, mk, Xp, yp, Xt, yt, K, rho, R, seed + 7)
                                     for n, mk in configs().items())
            for r_ in res:
                r_.update(dataset=ds, rho=rho, K=K, n_train=len(yp) // 2, n_test=nt); out.append(r_)
            print(ds, rho, round(time.time() - t0, 1), flush=True)
            json.dump(out, open("r1_results.json", "w"), indent=1)

if __name__ == "__main__":
    run(R=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
