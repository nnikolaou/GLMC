"""E1: scaled synthetic G-LMC experiment.
Adds: noise grid, MLP trained to interpolation, soft (probabilistic) outputs for
log-loss quantities and a test of Prop. G7 (held-out H(F|Y) ranks excess complexity).
Reuses the task of pilot.py (X=(Z,U), eta depends on Z only).
"""
import numpy as np, json, sys, time, warnings
from sklearn.neural_network import MLPClassifier
from pilot import sample_x, feats, eta_of_z, keys, h2, make_learners, NZ

warnings.filterwarnings("ignore")
QBINS = 20  # soft outputs quantized to multiples of 1/20

class MemSmoothP:
    def fit(self, z, u, y):
        self.table = dict(zip(keys(z, u).tolist(), y.tolist()))
        self.frac = np.array([y[z == k].mean() if np.any(z == k) else 0.5 for k in range(NZ)])
        return self
    def proba(self, z, u):
        out = self.frac[z].copy()
        for i, kk in enumerate(keys(z, u).tolist()):
            v = self.table.get(kk)
            if v is not None:
                out[i] = v
        return out

def H(*cols):
    a = np.stack(cols, 1)
    _, c = np.unique(a, return_counts=True, axis=0)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def learners(n):
    L = dict(make_learners(n))
    L.pop("MemSmooth (interp.)")
    L["MLP 512 (interp.)"] = lambda: ("sk", MLPClassifier((512,), alpha=0, max_iter=300, tol=0, n_iter_no_change=1000,
                                                         learning_rate_init=3e-3, batch_size=128))
    L = {"MemSmooth (interp.)": lambda: ("custom", MemSmoothP()), **L}
    return L

def run(eps_list, n_list, R, M, seed=1):
    rng = np.random.default_rng(seed)
    zt, ut = sample_x(rng, M); Xt = feats(zt, ut)
    rows = []
    for eps in eps_list:
        eta_t, b_t = eta_of_z(zt, eps)
        yt = (rng.random(M) < eta_t).astype(int)          # one held-out label draw per test input
        marg = np.abs(2 * eta_t - 1)
        HYX = float(np.mean(h2(eta_t)))
        etaq = np.round(eta_t * QBINS).astype(int)
        H_eta_given_Y = H(etaq, yt) - H(yt)
        for n in n_list:
            for name, mk in learners(n).items():
                t0 = time.time()
                hard = np.zeros((R, M), np.int8)
                rec = dict(tr=[], Efun01=[], Efun_log=[], Hfy=[], dlog=[])
                for r in range(R):
                    rr = np.random.default_rng(seed * 100000 + 1000 * r + n + int(eps * 100))
                    z, u = sample_x(rr, n); eta, _ = eta_of_z(z, eps)
                    y = (rr.random(n) < eta).astype(int)
                    kind, model = mk()
                    if kind == "custom":
                        model.fit(z, u, y); ptr = (model.proba(z, u) >= 0.5).astype(int); p = model.proba(zt, ut)
                    elif len(np.unique(y)) < 2:
                        ptr = np.full(n, y[0]); p = np.full(M, float(y[0]))
                    else:
                        X = feats(z, u); model.fit(X, y); ptr = model.predict(X); p = model.predict_proba(Xt)[:, 1]
                    f = (p >= 0.5).astype(int); hard[r] = f
                    fq = np.round(p * QBINS).astype(int)
                    rec["tr"].append(np.mean(ptr != y))
                    rec["Efun01"].append(H(f, b_t) - H(b_t))              # H(F|b*) hard outputs
                    rec["Efun_log"].append(H(fq, etaq) - H(etaq))         # H(F|eta) soft outputs (exact eta)
                    rec["Hfy"].append(H(fq, yt) - H(yt))                  # held-out H(F|Y), labels only
                    # log-loss deficit Delta = H(Y|F) - H(Y|X), exact via eta
                    hyf = 0.0
                    for v in np.unique(fq):
                        m = fq == v; hyf += m.mean() * h2(eta_t[m].mean())
                    rec["dlog"].append(hyf - HYX)
                p1 = hard.mean(0); pi = np.minimum(p1, 1 - p1); mode = (p1 >= 0.5).astype(int)
                V = np.where(pi > 0, h2(pi), 0.0)
                exc = float(np.mean(marg * np.mean(hard != b_t, 0)))
                bias = float(np.mean(marg * (mode != b_t)))
                row = dict(eps=eps, n=n, learner=name, R=R, H_YgX=HYX, H_eta_given_Y=H_eta_given_Y,
                           train_err=float(np.mean(rec["tr"])), excess_risk=exc, V=float(np.mean(V)), bias=bias,
                           G4_upper=float(0.5 * np.mean(marg * V) + bias),
                           E_fun01=float(np.mean(rec["Efun01"])), E_fun_log=float(np.mean(rec["Efun_log"])),
                           heldout_HFgY=float(np.mean(rec["Hfy"])),
                           G7_estimate=float(np.mean(rec["Hfy"]) - H_eta_given_Y),
                           logloss_deficit=float(np.mean(rec["dlog"])), secs=round(time.time() - t0, 1))
                rows.append(row); print(json.dumps(row), flush=True)
                json.dump(rows, open("e1_results.json", "w"), indent=1)
    return rows

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run([0.0, 0.1, 0.2, 0.3, 0.4], [256, 1024, 4096], R=R, M=20000)
