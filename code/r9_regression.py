"""R9: extension to regression. Protocol B analogue (one noisy dataset, matched compute) for real-valued targets.
Per case (dataset x noise level x seed): targets standardised; a clean test set (30%) keeps the original targets; the training
set (70%, at most NMAX points) gets one draw of Gaussian label noise N(0, s^2). For each of 17 regressors we record:
  refit on the full training set (test predictions -> regret target),
  holdout (80/20) predictions on the validation part,
  2-fold and 5-fold out-of-fold predictions (CV) and each fold model's predictions on the unlabelled test inputs,
  two further random 2-fold splits (test-input predictions; train-set predictions for CV) -> 3 disjoint pairs in total.
CPU time of every fit and prediction is recorded. Friedman #1 (synthetic, known regression function) is included as a dataset
whose 'clean' test targets are f(x), so its regret is exact excess risk.
Usage: python r9_regression.py [seed ...] [--batch-b]   (batch b: five further interpolating regressors, same data and splits)"""
import numpy as np, os, sys, time, warnings, pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_diabetes, make_friedman1
warnings.filterwarnings("ignore")
PMLB = os.environ.get("GLMC_PMLB_REG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pmlb_reg"))
REAL = ["1199_BNG_echoMonths", "215_2dplanes", "225_puma8NH", "197_cpu_act", "503_wind", "529_pollen", "537_houses", "564_fried", "344_mv"]
DATASETS = ["friedman1", "diabetes"] + REAL
NOISE = {"g0.5": 0.5, "g1.0": 1.0}
NMAX = 1000

class RandFeat:
    """Random ReLU features + minimum-norm least squares (interpolates when p >= n)."""
    def __init__(self, ratio=4.0, random_state=0): self.ratio, self.random_state = ratio, random_state
    def set_params(self, **kw): self.__dict__.update(kw); return self
    def fit(self, X, y):
        g = np.random.default_rng(self.random_state); p = max(1, int(self.ratio * len(y)))
        self.W = g.standard_normal((X.shape[1], p)) / np.sqrt(X.shape[1]); self.b = g.uniform(-1, 1, p)
        self.beta = np.linalg.pinv(np.maximum(X @ self.W + self.b, 0)) @ y; return self
    def predict(self, X): return np.maximum(X @ self.W + self.b, 0) @ self.beta

def configs(d):
    g = 1.0 / d
    C = {"kNN k=1": lambda: KNeighborsRegressor(1), "kNN k=5": lambda: KNeighborsRegressor(5), "kNN k=25": lambda: KNeighborsRegressor(25),
         "Tree depth=None": lambda: DecisionTreeRegressor(), "Tree depth=4": lambda: DecisionTreeRegressor(max_depth=4),
         "Tree depth=8": lambda: DecisionTreeRegressor(max_depth=8),
         "RF leaf=1": lambda: RandomForestRegressor(200, min_samples_leaf=1, n_jobs=1), "RF leaf=5": lambda: RandomForestRegressor(200, min_samples_leaf=5, n_jobs=1),
         "GB 300x d6 lr1": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=1.0),
         "GB 100x d3 lr.1": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1),
         "KRR ridgeless": lambda: KernelRidge(alpha=1e-8, kernel="rbf", gamma=g), "KRR alpha=0.1": lambda: KernelRidge(alpha=0.1, kernel="rbf", gamma=g),
         "KRR alpha=1": lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=g), "Ridge alpha=1": lambda: Ridge(1.0),
         "RF min-norm p=4n": lambda: RandFeat(4.0), "MLP512 alpha=1e-05": lambda: MLPRegressor(hidden_layer_sizes=(512,), alpha=1e-5, max_iter=300, tol=1e-6, n_iter_no_change=30),
         "MLP512 alpha=1": lambda: MLPRegressor(hidden_layer_sizes=(512,), alpha=1.0, max_iter=300)}
    return C

def configs_b(d):
    """Second batch (r9b): further interpolating regressors, so that the overparameterized pool contains several competitive ones."""
    g = 1.0 / d
    return {"kNN k=5 dist": lambda: KNeighborsRegressor(5, weights="distance"), "kNN k=25 dist": lambda: KNeighborsRegressor(25, weights="distance"),
            "ExtraTrees": lambda: ExtraTreesRegressor(200, n_jobs=1),
            "RF no-bootstrap mf=1/3": lambda: RandomForestRegressor(200, bootstrap=False, max_features=1 / 3, n_jobs=1),
            "KRR ridgeless Laplace": lambda: KernelRidge(alpha=1e-8, kernel="laplacian", gamma=g)}
OVER = ["kNN k=1", "Tree depth=None", "RF leaf=1", "GB 300x d6 lr1", "KRR ridgeless", "RF min-norm p=4n", "MLP512 alpha=1e-05",
        "kNN k=5 dist", "kNN k=25 dist", "ExtraTrees", "RF no-bootstrap mf=1/3", "KRR ridgeless Laplace"]   # designed to interpolate

def load(name, seed):
    rng = np.random.default_rng(seed)
    if name == "friedman1":
        X, _ = make_friedman1(n_samples=int(NMAX / 0.7) + 1, n_features=10, noise=0.0, random_state=seed)
        f = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]; y = f
    elif name == "diabetes":
        X, y = load_diabetes(return_X_y=True)
    else:
        df = pd.read_csv(os.path.join(PMLB, name, name + ".tsv.gz"), sep="\t"); y = df.pop("target").values.astype(float); X = df.values.astype(float)
        if len(y) > int(NMAX / 0.7) + 1:
            idx = rng.choice(len(y), int(NMAX / 0.7) + 1, replace=False); X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X); y = (y - y.mean()) / y.std()
    return X, y

def per_config(name, mk, Xtr, ytr, Xte, folds, extra, hold, seed):
    out = {"name": name}; t = {}
    def fit(Xa, ya, s):
        m = mk()
        if hasattr(m, "random_state"): m.set_params(random_state=s)
        return m.fit(Xa, ya)
    t0 = time.process_time(); m = fit(Xtr, ytr, seed); out["test_refit"] = m.predict(Xte).astype(np.float32); t["refit"] = time.process_time() - t0
    a, b = hold
    t0 = time.process_time(); m = fit(Xtr[a], ytr[a], seed + 1); out["hold_pred"] = m.predict(Xtr[b]).astype(np.float32); t["holdout"] = time.process_time() - t0
    for K, fl in folds.items():
        oof = np.zeros(len(ytr), np.float32); tp = []; tcv = tte = 0.0
        for k in range(K):
            tr = fl != k
            t0 = time.process_time(); m = fit(Xtr[tr], ytr[tr], seed + 10 * K + k); oof[~tr] = m.predict(Xtr[~tr]); tcv += time.process_time() - t0
            t0 = time.process_time(); tp.append(m.predict(Xte)); tte += time.process_time() - t0
        out[f"oof{K}"] = oof; out[f"testfold{K}"] = np.array(tp, np.float32); t[f"cv{K}"] = tcv; t[f"unlab{K}"] = tte
    ex_te, ex_oof = [], []; tcv = tte = 0.0
    for d, fl in enumerate(extra):                                     # two further random 2-fold splits
        oof = np.zeros(len(ytr), np.float32); tp = []
        for k in range(2):
            tr = fl != k
            t0 = time.process_time(); m = fit(Xtr[tr], ytr[tr], seed + 1000 * (d + 1) + k); oof[~tr] = m.predict(Xtr[~tr]); tcv += time.process_time() - t0
            t0 = time.process_time(); tp.append(m.predict(Xte)); tte += time.process_time() - t0
        ex_te.append(tp); ex_oof.append(oof)
    out["extra_test"] = np.array(ex_te, np.float32); out["extra_oof"] = np.array(ex_oof, np.float32); t["extra"] = tcv; t["unlab_extra"] = tte
    out["times"] = t; return out

def run_case(ds, nz, seed, batch="a"):
    f = f"r9_{ds}_{nz}_s{seed}.npz" if batch == "a" else f"r9b_{ds}_{nz}_s{seed}.npz"
    if os.path.exists(f): return
    X, y = load(ds, seed); rng = np.random.default_rng(seed + 7)
    idx = rng.permutation(len(y)); nt = int(0.3 * len(y)); te, tr = idx[:nt], idx[nt:nt + NMAX]
    ytr = y[tr] + NOISE[nz] * rng.standard_normal(len(tr))
    hp = rng.permutation(len(tr)); cut = int(0.8 * len(tr)); hold = (hp[:cut], hp[cut:])
    folds = {K: rng.permutation(np.arange(len(tr)) % K) for K in (2, 5)}
    extra = [rng.permutation(np.arange(len(tr)) % 2) for _ in range(2)]
    t0 = time.time()
    res = Parallel(n_jobs=2)(delayed(per_config)(n, mk, X[tr], ytr, X[te], folds, extra, hold, seed + 100) for n, mk in (configs if batch == "a" else configs_b)(X.shape[1]).items())
    np.savez_compressed(f, names=np.array([r["name"] for r in res]), yte=y[te], ytr_noisy=ytr, ytr_clean=y[tr], sigma=NOISE[nz],
                        hold_idx=hold[1], fold2=folds[2], fold5=folds[5], extra=np.array(extra),
                        test_refit=np.stack([r["test_refit"] for r in res]), hold_pred=np.stack([r["hold_pred"] for r in res]),
                        oof2=np.stack([r["oof2"] for r in res]), oof5=np.stack([r["oof5"] for r in res]),
                        testfold2=np.stack([r["testfold2"] for r in res]), testfold5=np.stack([r["testfold5"] for r in res]),
                        extra_test=np.stack([r["extra_test"] for r in res]), extra_oof=np.stack([r["extra_oof"] for r in res]),
                        times=np.array([[r["times"][k] for k in ("refit", "holdout", "cv2", "cv5", "unlab2", "unlab5", "extra", "unlab_extra")] for r in res]))
    print(ds, nz, seed, round(time.time() - t0, 1), flush=True)

if __name__ == "__main__":
    args = sys.argv[1:]; batch = "b" if "--batch-b" in args else "a"
    seeds = [int(s) for s in args if s.isdigit()] or [0, 1, 2, 3, 4]
    for seed in seeds:
        for ds in DATASETS:
            for nz in NOISE:
                run_case(ds, nz, seed, batch)
