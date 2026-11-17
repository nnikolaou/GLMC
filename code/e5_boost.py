"""E5-lite: boosting trajectories, old information plane vs G-LMC quantities, with and without label noise."""
import numpy as np, json, sys
from sklearn.ensemble import GradientBoostingClassifier
from pilot import sample_x, feats, eta_of_z, h2
from e1 import H, QBINS

def run(n=1024, R=30, M=20000, T=200, every=5, seed=7):
    rng = np.random.default_rng(seed)
    zt, ut = sample_x(rng, M); Xt = feats(zt, ut)
    steps = list(range(every - 1, T, every))
    out = []
    for eps in [0.0, 0.2]:
        eta_t, b_t = eta_of_z(zt, eps); marg = np.abs(2 * eta_t - 1)
        yt = (rng.random(M) < eta_t).astype(int); etaq = np.round(eta_t * QBINS).astype(int)
        for lr in [1.0, 0.1]:
            hard = np.zeros((R, len(steps), M), np.int8)
            acc = {k: np.zeros((R, len(steps))) for k in ["Efun01", "Efun_log", "Hfy", "leg_IFY", "leg_IFX", "train_err"]}
            for r in range(R):
                rr = np.random.default_rng(seed * 1000 + r)
                z, u = sample_x(rr, n); eta, _ = eta_of_z(z, eps); y = (rr.random(n) < eta).astype(int)
                X = feats(z, u)
                m = GradientBoostingClassifier(loss="exponential", learning_rate=lr, n_estimators=T, max_depth=6).fit(X, y)
                tr = list(m.staged_predict_proba(X)); te = list(m.staged_predict_proba(Xt))
                for j, s in enumerate(steps):
                    p = te[s][:, 1]; f = (p >= 0.5).astype(int); hard[r, j] = f; fq = np.round(p * QBINS).astype(int)
                    acc["Efun01"][r, j] = H(f, b_t) - H(b_t)
                    acc["Efun_log"][r, j] = H(fq, etaq) - H(etaq)
                    acc["Hfy"][r, j] = H(fq, yt) - H(yt)
                    ptr = tr[s][:, 1]
                    acc["train_err"][r, j] = np.mean((ptr >= 0.5) != y)
                    # legacy information plane (old paper): scores binned into 100 bins on training data
                    fb = np.minimum((ptr * 100).astype(int), 99)
                    Hy = H(y); acc["leg_IFY"][r, j] = (H(fb) + Hy - H(fb, y)) / Hy
                    acc["leg_IFX"][r, j] = H(fb) / np.log2(n)
            for j, s in enumerate(steps):
                p1 = hard[:, j].mean(0); pi = np.minimum(p1, 1 - p1); mode = (p1 >= 0.5).astype(int)
                V = np.where(pi > 0, h2(pi), 0.0)
                out.append(dict(eps=eps, lr=lr, round=s + 1,
                                excess_risk=float(np.mean(marg * np.mean(hard[:, j] != b_t, 0))),
                                V=float(V.mean()), bias=float(np.mean(marg * (mode != b_t))),
                                **{k: float(v[:, j].mean()) for k, v in acc.items()}))
            print(eps, lr, json.dumps(out[-1]), flush=True)
            json.dump(out, open("e5_results.json", "w"), indent=1)

if __name__ == "__main__":
    run(R=int(sys.argv[1]) if len(sys.argv) > 1 else 30)
