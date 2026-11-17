"""G-LMC pilot: stored vs expressed excess information on a synthetic task
with exactly known eta(x), Bayes classifier b*(x) and noise entropy H(Y|X).

X = (Z, U): Z in {0..7} (relevant), U in {0,1}^d (nuisance). eta depends on Z only.
For each (noise eps, n, learner) we train R models on independent training sets and
evaluate on a fixed Monte-Carlo sample of M test inputs (exact expectation over Y).
"""
import numpy as np, json, sys, time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

D_NUIS = 16
NZ = 8
LOG2 = np.log(2)

def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def eta_of_z(z, eps):
    bstar = (z >= 4).astype(int)
    ez = np.where(z % 2 == 0, eps, eps / 2)
    return np.where(bstar == 1, 1 - ez, ez), bstar

def sample_x(rng, m):
    z = rng.integers(0, NZ, m)
    u = rng.integers(0, 2, (m, D_NUIS))
    return z, u

def feats(z, u):
    oh = np.eye(NZ)[z]
    return np.hstack([oh, u]).astype(np.float32)

def keys(z, u):
    w = (1 << np.arange(D_NUIS)).astype(np.int64)
    return z.astype(np.int64) * (1 << D_NUIS) + u.astype(np.int64) @ w

class MemSmooth:
    """Interpolator: exact-match lookup, else majority label among same-Z training points."""
    def fit(self, z, u, y):
        self.table = dict(zip(keys(z, u).tolist(), y.tolist()))  # last label wins on dup keys
        self.maj = np.array([(y[z == k].mean() >= 0.5) if np.any(z == k) else 1 for k in range(NZ)]).astype(int)
        return self
    def predict(self, z, u):
        k = keys(z, u).tolist()
        out = self.maj[z].copy()
        for i, kk in enumerate(k):
            v = self.table.get(kk)
            if v is not None:
                out[i] = v
        return out

def make_learners(n):
    k = int(np.sqrt(n)) | 1
    return {
        "MemSmooth (interp.)": lambda: ("custom", MemSmooth()),
        "1-NN": lambda: ("sk", KNeighborsClassifier(1, metric="hamming", algorithm="brute")),
        f"k-NN (k=sqrt n)": lambda: ("sk", KNeighborsClassifier(k, metric="hamming", algorithm="brute")),
        "Tree (full depth)": lambda: ("sk", DecisionTreeClassifier()),
        "Random forest": lambda: ("sk", RandomForestClassifier(100, n_jobs=2)),
        "GBoost (exp, lr=1, d=6)": lambda: ("sk", GradientBoostingClassifier(loss="exponential", learning_rate=1.0,
                                                                             n_estimators=100, max_depth=6)),
        "Logistic (C=1e4)": lambda: ("sk", LogisticRegression(C=1e4, max_iter=2000)),
    }

def emp_mi_metrics(f, y):
    """Legacy empirical quantities on the training set (hard outputs, all x distinct)."""
    def H(a):
        _, c = np.unique(a, return_counts=True, axis=0)
        p = c / c.sum(); return -(p * np.log2(p)).sum()
    Hy, Hf = H(y), H(f)
    Hfy = H(np.stack([f, y], 1))
    Ify = Hf + Hy - Hfy
    return (Ify / Hy if Hy > 0 else 1.0), (Hf / Hy if Hy > 0 else 1.0)  # I_S(F;Y)/H_S(Y), I_S(F;X)/H_S(Y) (=H_S(F)/H_S(Y))

def run(eps_list, n_list, R, M, seed=0):
    rng = np.random.default_rng(seed)
    zt, ut = sample_x(rng, M)
    Xt = feats(zt, ut)
    rows = []
    for eps in eps_list:
        eta_t, b_t = eta_of_z(zt, eps)
        marg = np.abs(2 * eta_t - 1)
        HYX = float(np.mean(h2(eta_of_z(np.arange(NZ), eps)[0]))) if eps > 0 else 0.0
        for n in n_list:
            for name, mk in make_learners(n).items():
                t0 = time.time()
                preds = np.zeros((R, M), dtype=np.int8)
                tr_err, legacy_ify, legacy_hf = [], [], []
                for r in range(R):
                    rr = np.random.default_rng(10_000 * seed + 1000 * r + n + int(eps * 100))
                    z, u = sample_x(rr, n)
                    eta, _ = eta_of_z(z, eps)
                    y = (rr.random(n) < eta).astype(int)
                    kind, model = mk()
                    if kind == "custom":
                        model.fit(z, u, y); ptr = model.predict(z, u); preds[r] = model.predict(zt, ut)
                    else:
                        X = feats(z, u)
                        if len(np.unique(y)) < 2:
                            ptr = np.full(n, y[0]); preds[r] = y[0]
                        else:
                            model.fit(X, y); ptr = model.predict(X); preds[r] = model.predict(Xt)
                    tr_err.append(np.mean(ptr != y))
                    a, b = emp_mi_metrics(ptr, y); legacy_ify.append(a); legacy_hf.append(b)
                p1 = preds.mean(0)                       # P_S(f_W(x')=1)
                mode = (p1 >= 0.5).astype(int)
                pi = np.minimum(p1, 1 - p1)
                V = h2(pi); V[pi < 1e-12] = 0.0          # expressed info per test point (bits)
                exc = np.mean(marg * np.mean(preds != b_t, axis=0))       # E_S R - R*
                bias = np.mean(marg * (mode != b_t))
                bound = 0.5 * np.mean(marg * V) + bias
                # function-level decision excess complexity H(F | b*(X)) averaged over S
                Efun = []
                for r in range(R):
                    f = preds[r]
                    hc = 0.0
                    for bv in (0, 1):
                        msk = b_t == bv
                        if msk.any():
                            hc += msk.mean() * h2(f[msk].mean()) if 0 < f[msk].mean() < 1 else 0.0
                    Efun.append(hc)
                ebar = float(np.mean(tr_err))
                stored_lb = max(0.0, HYX - (h2(ebar) if ebar > 0 else 0.0))  # per-sample Fano lower bound (bits)
                rows.append(dict(eps=eps, n=n, learner=name, H_YgX=HYX, train_err=ebar,
                                 excess_risk=float(exc), V=float(np.mean(V)), bias=float(bias),
                                 G4_bound=float(bound), E_fun=float(np.mean(Efun)),
                                 stored_per_sample_lb=float(stored_lb),
                                 legacy_IFY_over_HY=float(np.mean(legacy_ify)),
                                 legacy_IFX_over_HY=float(np.mean(legacy_hf)),
                                 secs=round(time.time() - t0, 1)))
                print(json.dumps(rows[-1]), flush=True)
    return rows

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rows = run(eps_list=[0.0, 0.2, 0.4], n_list=[256, 1024, 4096], R=R, M=20000)
    json.dump(rows, open("results.json", "w"), indent=1)
