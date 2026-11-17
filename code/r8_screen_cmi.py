"""R8: data for (i) two-stage selection (screen by TRAINING fit, then min-V) and (ii) CMI-based selection, on Protocol B.

For every Protocol B case (15 datasets x 3 noise models x 5 seeds; data, noise and 2-fold split reproduced exactly as in
r6_cost.py) and each of the 19 configurations, we fit D = 3 antithetic pairs of half-sample models on a FIXED supersample:
  * the training set is split into pairs (a_i, b_i), a_i from fold 0 and b_i from fold 1 of the Protocol B 2-fold split
    (random matching; an odd leftover point is always added to the first model of each pair);
  * draw d has a selection vector U^d in {0,1}^{pairs}; model (d,0) is trained on {a_i if U_i=0 else b_i}, model (d,1) on the
    complement. Draw 0 has U = 1 (model (0,0) on fold 1, model (0,1) on fold 0), i.e. it is exactly the Protocol B 2-fold split (same seeds, so it reproduces oof2/testfold2);
    draws 1 and 2 have i.i.d. uniform U.
Each model's predictions are stored on ALL training points (own half -> training error, other half -> held-out loss, both ->
supersample patterns for CMI) and on the test inputs (unlabelled inputs for V). CPU time per fit is recorded.
Usage: python r8_screen_cmi.py [seed ...]   (skips cases already on disk)"""
import numpy as np, sys, os, time, warnings
from joblib import Parallel, delayed
from r1_real import configs
from r6_cost import load_any, corrupt, NEW, OLD
warnings.filterwarnings("ignore")
D = 3

def setup(ds, kind, seed):
    X, y = load_any(ds, seed=seed); K = int(y.max()) + 1
    rng = np.random.default_rng(seed + 7); idx = rng.permutation(len(y)); nt = int(0.3 * len(y))
    te, tr = idx[:nt], idx[nt:]
    ytr = corrupt(X[tr], y[tr], kind, K, rng, Xref=X, yref=y)
    hp = rng.permutation(len(tr))                                         # holdout split (unused here, keeps the rng in step)
    folds = {Kf: rng.permutation(np.arange(len(tr)) % Kf) for Kf in (2, 5)}
    f2 = folds[2]; r2 = np.random.default_rng(seed + 999)
    i0, i1 = r2.permutation(np.where(f2 == 0)[0]), r2.permutation(np.where(f2 == 1)[0])
    m = min(len(i0), len(i1)); pairs = np.stack([i0[:m], i1[:m]], 1); left = np.concatenate([i0[m:], i1[m:]])
    U = np.ones((D, m), int); U[1:] = r2.integers(0, 2, (D - 1, m))
    return X[tr], ytr, y[tr], X[te], y[te], K, f2, pairs, left, U

def per_config(name, mk, Xtr, ytr, Xte, pairs, left, U, f2, seed):
    out = {"name": name}; ptr = np.zeros((D, 2, len(ytr)), np.int16); pte = np.zeros((D, 2, len(Xte)), np.int16); t = np.zeros((D, 2))
    m = pairs.shape[0]; ar = np.arange(m)
    for d in range(D):
        sideA = pairs[ar, U[d]]; sideB = pairs[ar, 1 - U[d]]
        sets = (np.concatenate([sideA, left]), sideB)
        if d == 0: sets = (np.where(f2 == 1)[0], np.where(f2 == 0)[0])   # draw 0 = the Protocol B 2-fold split, exactly as r6_cost.py
        for s, tr_idx in enumerate(sets):
            mdl = mk(); fs = seed + 20 + s if d == 0 else seed + 1000 * d + s
            if hasattr(mdl, "random_state"): mdl.set_params(random_state=fs)
            t0 = time.process_time(); mdl.fit(Xtr[tr_idx], ytr[tr_idx]); t[d, s] = time.process_time() - t0
            ptr[d, s] = mdl.predict(Xtr); pte[d, s] = mdl.predict(Xte)
    out.update(ptr=ptr, pte=pte, t=t); return out

def run_case(ds, kind, seed):
    f = f"r8_{ds}_{kind}_s{seed}.npz"
    if os.path.exists(f): return
    Xtr, ytr, yc, Xte, yte, K, f2, pairs, left, U = setup(ds, kind, seed)
    t0 = time.time()
    res = Parallel(n_jobs=2)(delayed(per_config)(n, mk, Xtr, ytr, Xte, pairs, left, U, f2, seed + 100) for n, mk in configs().items())
    np.savez_compressed(f, names=np.array([r["name"] for r in res]), K=K, ytr_noisy=ytr, ytr_clean=yc, yte=yte, fold2=f2,
                        pairs=pairs, left=left, U=U, ptr=np.stack([r["ptr"] for r in res]), pte=np.stack([r["pte"] for r in res]),
                        times=np.stack([r["t"] for r in res]))
    print(ds, kind, seed, round(time.time() - t0, 1), flush=True)

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
    for seed in seeds:
        for ds in NEW + OLD:
            for kind in ("sym0.2", "classdep", "instdep"):
                run_case(ds, kind, seed)
