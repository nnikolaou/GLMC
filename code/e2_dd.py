"""E2: double descent in a random-features (ReLU) least-squares classifier on the synthetic G-LMC task.
Sweeps width p around n; min-norm (lambda=0) vs small ridge. Measures excess risk, V, bias, train error.
Randomness of the features is part of the learner's internal seed xi (so V counts it)."""
import numpy as np, json, sys, time
from pilot import sample_x, feats, eta_of_z, h2

def fit_predict(Phi, y, Phit, lam):
    n, p = Phi.shape
    if lam == 0:
        theta = np.linalg.lstsq(Phi, y, rcond=None)[0]           # min-norm solution
    elif p >= n:
        theta = Phi.T @ np.linalg.solve(Phi @ Phi.T + lam * n * np.eye(n), y)
    else:
        theta = np.linalg.solve(Phi.T @ Phi + lam * n * np.eye(p), Phi.T @ y)
    return Phi @ theta, Phit @ theta

def run(n=512, ps=(16, 32, 64, 128, 256, 384, 448, 496, 512, 528, 576, 640, 768, 1024, 2048, 4096),
        lams=(0.0, 1e-2), eps_list=(0.0, 0.2), R=40, M=10000, seed=3):
    rng = np.random.default_rng(seed)
    zt, ut = sample_x(rng, M); Xt = feats(zt, ut).astype(np.float64)
    d = Xt.shape[1]
    rows = []
    for eps in eps_list:
        eta_t, b_t = eta_of_z(zt, eps); marg = np.abs(2 * eta_t - 1)
        for lam in lams:
            for p in ps:
                t0 = time.time()
                hard = np.zeros((R, M), np.int8); tr = []
                for r in range(R):
                    rr = np.random.default_rng(seed * 10**6 + 1000 * r + int(eps * 100))
                    z, u = sample_x(rr, n); eta, _ = eta_of_z(z, eps)
                    y = (rr.random(n) < eta).astype(int); ys = 2.0 * y - 1
                    W = rr.normal(size=(d, p)) / np.sqrt(d); b = rr.normal(size=p) * 0.5
                    Phi = np.maximum(feats(z, u) @ W + b, 0) / np.sqrt(p)
                    Phit = np.maximum(Xt @ W + b, 0) / np.sqrt(p)
                    ftr, fte = fit_predict(Phi, ys, Phit, lam)
                    tr.append(np.mean((ftr > 0) != y)); hard[r] = (fte > 0)
                p1 = hard.mean(0); pi = np.minimum(p1, 1 - p1); mode = (p1 >= 0.5).astype(int)
                V = np.where(pi > 0, h2(pi), 0.0)
                row = dict(eps=eps, lam=lam, n=n, p=p, ratio=p / n, train_err=float(np.mean(tr)),
                           excess_risk=float(np.mean(marg * np.mean(hard != b_t, 0))), V=float(V.mean()),
                           bias=float(np.mean(marg * (mode != b_t))), secs=round(time.time() - t0, 1))
                rows.append(row); print(json.dumps(row), flush=True)
                json.dump(rows, open("e2_results.json", "w"), indent=1)

if __name__ == "__main__":
    run(R=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
