"""R4: model selection — holdout validation vs expressed information V vs combinations.
Splits each dataset into a clean test set (30%, labels used only for scoring; its inputs double as the
unlabeled pool for V), a validation set (20%, NOISY labels, same noise process as training) and a training
pool (50%). Each of R runs trains on a random half of the pool with fresh noisy labels and a fresh seed.
Noise types: symmetric 0.2 and 0.3; class-dependent (binary: 0->1 w.p. 0.4, 1->0 w.p. 0.05;
multiclass: k->k+1 mod K w.p. 0.35). Stores all predictions so selection can be analysed offline."""
import numpy as np, sys, time, warnings
from joblib import Parallel, delayed
from realdata import load, DATASETS
from r1_real import configs
warnings.filterwarnings("ignore")

def corrupt(y, kind, K, rng):
    y = y.copy()
    if kind.startswith("sym"):
        rho = float(kind[3:]); m = rng.random(len(y)) < rho
        y[m] = (y[m] + rng.integers(1, K, m.sum())) % K
    elif kind == "classdep":
        if K == 2:
            u = rng.random(len(y)); f = ((y == 0) & (u < 0.4)) | ((y == 1) & (u < 0.05)); y[f] = 1 - y[f]
        else:
            m = rng.random(len(y)) < 0.35; y[m] = (y[m] + 1) % K
    return y

def one(name, mk, Xp, yp, Xv, Xt, K, kind, R, seed):
    Pv = np.zeros((R, len(Xv)), np.int16); Pt = np.zeros((R, len(Xt)), np.int16); tr = []
    for r in range(R):
        rng = np.random.default_rng(seed * 1000 + r)
        a = rng.permutation(len(yp))[: len(yp) // 2]; yn = corrupt(yp, kind, K, rng)
        m = mk()
        if hasattr(m, "random_state"): m.set_params(random_state=int(rng.integers(1e9)))
        m.fit(Xp[a], yn[a]); Pv[r] = m.predict(Xv); Pt[r] = m.predict(Xt); tr.append(np.mean(m.predict(Xp[a]) != yn[a]))
    return name, Pv, Pt, float(np.mean(tr))

def run(R=12, kinds=("sym0.2", "sym0.3", "classdep"), seed=0):
    for ds in DATASETS:
        X, y = load(ds, seed); K = int(y.max()) + 1
        rng = np.random.default_rng(seed + 1); idx = rng.permutation(len(y))
        nt, nv = int(0.3 * len(y)), int(0.2 * len(y))
        it, iv, ip = idx[:nt], idx[nt:nt + nv], idx[nt + nv:]
        for kind in kinds:
            t0 = time.time()
            yv_noisy = corrupt(y[iv], kind, K, np.random.default_rng(seed + 99))  # one fixed noisy validation labelling
            res = Parallel(n_jobs=2)(delayed(one)(n, mk, X[ip], y[ip], X[iv], X[it], K, kind, R, seed + 5)
                                     for n, mk in configs().items())
            np.savez_compressed(f"r4_{ds}_{kind}_s{seed}.npz", names=np.array([r[0] for r in res]),
                                Pv=np.stack([r[1] for r in res]), Pt=np.stack([r[2] for r in res]),
                                train_err=np.array([r[3] for r in res]), yv=yv_noisy, yv_clean=y[iv], yt=y[it], K=K)
            print(ds, kind, round(time.time() - t0, 1), flush=True)

if __name__ == "__main__":
    run(R=int(sys.argv[1]) if len(sys.argv) > 1 else 12, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
