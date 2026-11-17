"""R9 exact check (regression): Friedman #1 with known regression function f*, independent training samples.
For each regressor and noise level: exact excess risk E||f_W - f*||^2, squared bias ||m - f*||^2 and variance E v(X), estimated from
R independent training sets of size n (plug-in with ddof=0, so Exc = Bias^2 + Var holds exactly for the estimates), plus the
two-halves variance estimate V2 = mean (f_1 - f_2)^2 / 2 from one dataset of size n split into disjoint halves."""
import numpy as np, json, sys, warnings
from sklearn.datasets import make_friedman1
from r9_regression import configs, configs_b, OVER
warnings.filterwarnings("ignore")

def f_star(X): return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]

def run(n=500, R=10, M=4000, sigmas=(0.5, 1.0), seed=0, batch="a"):
    rng = np.random.default_rng(seed)
    Xt = rng.uniform(size=(M, 10)); ft = f_star(Xt); mu, sd = ft.mean(), ft.std(); ft = (ft - mu) / sd   # standardised scale
    rows = []
    for s in sigmas:
        for name, mk in (configs if batch == "a" else configs_b)(10).items():
            P = np.zeros((R, M)); P2 = np.zeros((2, M)); tr_mse = []
            for r in range(R):
                g = np.random.default_rng(1000 * r + int(100 * s) + seed)
                X = g.uniform(size=(n, 10)); y = (f_star(X) - mu) / sd + s * g.standard_normal(n)
                m = mk()
                if hasattr(m, "random_state"): m.set_params(random_state=r)
                m.fit(X, y); P[r] = m.predict(Xt); tr_mse.append(float(np.mean((m.predict(X) - y) ** 2)))
                if r == 0:                                                 # two disjoint halves of the first dataset
                    for h in range(2):
                        mh = mk()
                        if hasattr(mh, "random_state"): mh.set_params(random_state=50 + h)
                        mh.fit(X[h::2], y[h::2]); P2[h] = mh.predict(Xt)
            mean = P.mean(0); var = P.var(0)
            row = dict(sigma=s, learner=name, over=name in OVER, train_mse=float(np.mean(tr_mse)), excess=float(np.mean((P - ft) ** 2)),
                       bias2=float(np.mean((mean - ft) ** 2)), var=float(var.mean()), V2=float(np.mean((P2[0] - P2[1]) ** 2) / 2))
            rows.append(row); print(json.dumps(row), flush=True)
    json.dump(rows, open("r9_exact.json" if batch == "a" else "r9_exact_b.json", "w"), indent=1)

if __name__ == "__main__":
    run(batch="b" if "--batch-b" in sys.argv else "a")
