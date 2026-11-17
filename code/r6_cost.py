"""R6: realistic, cost-matched model selection on one noisy training set.
Per case (dataset x noise): clean test set (30%), single noisy training set (70%). For each of 19 configurations:
  refit on the full training set (scored on clean test -> the regret target),
  1-fit holdout (train 80% / validate 20% of the training set),
  K-fold for K in (2, 5): out-of-fold predictions (CV error, T-hat) and each fold model's test-input predictions (V, consensus).
Wall-clock time of every fit is recorded so criteria can be compared at equal compute.
Noise: symmetric 0.2; class-dependent (as R4); instance-dependent (flip w.p. min(0.5, 0.7 u(x)) to the reference model's
second most likely class, where u(x) = 1 - (p1 - p2) from a reference model fit on clean data)."""
import numpy as np, sys, time, os, warnings, pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from r1_real import configs
from realdata import load as load_small
warnings.filterwarnings("ignore")
PMLB = os.environ.get("GLMC_PMLB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pmlb"))
NEW = ["twonorm", "ring", "magic", "phoneme", "spambase", "churn", "waveform_40", "satimage", "optdigits", "kr_vs_kp"]
OLD = ["wdbc", "biopsy", "oj", "caravan", "digits"]

def load_any(name, nmax=2000, seed=0):
    if name in OLD:
        X, y = load_small(name, seed)
    else:
        df = pd.read_csv(os.path.join(PMLB, name, name + ".tsv.gz"), sep="\t", compression="gzip")
        y = df.pop("target").values; X = df.values.astype(float)
        _, y = np.unique(y, return_inverse=True)
        X = StandardScaler().fit_transform(X).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(y) > nmax:
        idx = rng.choice(len(y), nmax, replace=False); X, y = X[idx], y[idx]
    _, y = np.unique(y, return_inverse=True)
    return X, y.astype(int)

def corrupt(X, y, kind, K, rng, Xref=None, yref=None):
    y = y.copy()
    if kind == "sym0.2":
        m = rng.random(len(y)) < 0.2; y[m] = (y[m] + rng.integers(1, K, m.sum())) % K
    elif kind == "classdep":
        if K == 2:
            u = rng.random(len(y)); f = ((y == 0) & (u < 0.4)) | ((y == 1) & (u < 0.05)); y[f] = 1 - y[f]
        else:
            m = rng.random(len(y)) < 0.35; y[m] = (y[m] + 1) % K
    elif kind == "instdep":
        ref = RandomForestClassifier(300, min_samples_leaf=3, random_state=0).fit(Xref, yref)
        P = np.zeros((len(y), K)); P[:, ref.classes_] = ref.predict_proba(X)
        srt = np.sort(P, 1); u = 1 - (srt[:, -1] - srt[:, -2])
        second = np.argsort(P, 1)[:, -2]
        top = np.argsort(P, 1)[:, -1]
        alt = np.where(second == y, top, second)            # flip to the most confusable other class
        m = rng.random(len(y)) < np.minimum(0.5, 0.7 * u); y[m] = alt[m]
    return y

def per_config(name, mk, Xtr, ytr, Xte, folds, hold, seed):
    """Times include fitting AND the predictions each protocol needs."""
    out = {"name": name}; t = {}
    def fit(Xa, ya, s):
        m = mk()
        if hasattr(m, "random_state"): m.set_params(random_state=s)
        return m.fit(Xa, ya)
    t0 = time.process_time(); m = fit(Xtr, ytr, seed); out["test_refit"] = m.predict(Xte).astype(np.int16); t["refit"] = time.process_time() - t0
    a, b = hold
    t0 = time.process_time(); m = fit(Xtr[a], ytr[a], seed + 1); out["hold_pred"] = m.predict(Xtr[b]).astype(np.int16); t["holdout"] = time.process_time() - t0
    for K, fl in folds.items():
        oof = np.zeros(len(ytr), np.int16); tp = []; tcv = 0.0; tte = 0.0
        for k in range(K):
            tr = fl != k
            t0 = time.process_time(); m = fit(Xtr[tr], ytr[tr], seed + 10 * K + k); oof[~tr] = m.predict(Xtr[~tr]); tcv += time.process_time() - t0
            t0 = time.process_time(); tp.append(m.predict(Xte)); tte += time.process_time() - t0
        out[f"oof{K}"] = oof; out[f"testfold{K}"] = np.array(tp, np.int16); t[f"cv{K}"] = tcv; t[f"unlab{K}"] = tte
    out["times"] = t
    return out

def run_case(ds, kind, seed=0):
    f = f"r6_{ds}_{kind}_s{seed}.npz"
    if os.path.exists(f): return
    X, y = load_any(ds, seed=seed); K = int(y.max()) + 1
    rng = np.random.default_rng(seed + 7); idx = rng.permutation(len(y)); nt = int(0.3 * len(y))
    te, tr = idx[:nt], idx[nt:]
    ytr = corrupt(X[tr], y[tr], kind, K, rng, Xref=X, yref=y)
    hp = rng.permutation(len(tr)); cut = int(0.8 * len(tr)); hold = (hp[:cut], hp[cut:])
    folds = {Kf: rng.permutation(np.arange(len(tr)) % Kf) for Kf in (2, 5)}
    t0 = time.time()
    res = Parallel(n_jobs=2)(delayed(per_config)(n, mk, X[tr], ytr, X[te], folds, hold, seed + 100) for n, mk in configs().items())
    np.savez_compressed(f, names=np.array([r["name"] for r in res]), K=K, yte=y[te], ytr_noisy=ytr, ytr_clean=y[tr],
                        hold_idx=hold[1], fold2=folds[2], fold5=folds[5],
                        test_refit=np.stack([r["test_refit"] for r in res]), hold_pred=np.stack([r["hold_pred"] for r in res]),
                        oof2=np.stack([r["oof2"] for r in res]), oof5=np.stack([r["oof5"] for r in res]),
                        testfold2=np.stack([r["testfold2"] for r in res]), testfold5=np.stack([r["testfold5"] for r in res]),
                        times=np.array([[r["times"][k] for k in ("refit", "holdout", "cv2", "cv5", "unlab2", "unlab5")] for r in res]))
    print(ds, kind, round(time.time() - t0, 1), flush=True)

if __name__ == "__main__":
    for ds in NEW + OLD:
        for kind in ("sym0.2", "classdep", "instdep"):
            run_case(ds, kind)
