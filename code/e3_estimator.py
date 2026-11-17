"""E3: bias of the expressed-information estimator V as a function of the number R of independent training runs.
Reference: plug-in V from R_ref = 200 runs. Compare plug-in vs Miller-Madow corrected estimates for R in {5,...,100}."""
import numpy as np, json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pilot import sample_x, feats, eta_of_z, h2

LEARNERS = {"1-NN": lambda: KNeighborsClassifier(1, metric="hamming", algorithm="brute"),
            "Full-depth tree": lambda: DecisionTreeClassifier(),
            "Random forest": lambda: RandomForestClassifier(50, n_jobs=1),
            "Logistic": lambda: LogisticRegression(C=1e4, max_iter=2000)}

def V_est(P, mm):
    p1 = P.mean(0); R = P.shape[0]
    H = np.where((p1 > 0) & (p1 < 1), h2(np.clip(p1, 1e-12, 1 - 1e-12)), 0.0)
    if mm:  # Miller-Madow: + (k-1)/(2R) nats, k = number of observed values (1 or 2)
        k = 1 + ((p1 > 0) & (p1 < 1))
        H = H + (k - 1) / (2 * R * np.log(2))
    return float(H.mean())

def run(n=1024, eps=0.2, Rref=200, M=5000, Rs=(5, 10, 20, 40, 80, 100), reps=20, seed=11):
    rng = np.random.default_rng(seed)
    zt, ut = sample_x(rng, M); Xt = feats(zt, ut)
    out = []
    for name, mk in LEARNERS.items():
        P = np.zeros((Rref, M), np.int8)
        for r in range(Rref):
            rr = np.random.default_rng(seed * 10**5 + r)
            z, u = sample_x(rr, n); eta, _ = eta_of_z(z, eps); y = (rr.random(n) < eta).astype(int)
            P[r] = mk().fit(feats(z, u), y).predict(Xt)
        ref = V_est(P, False)
        for R in Rs:
            pl, mm = [], []
            for k in range(reps):
                idx = np.random.default_rng(k).choice(Rref, R, replace=False)
                pl.append(V_est(P[idx], False)); mm.append(V_est(P[idx], True))
            out.append(dict(learner=name, R=R, V_ref=ref, plugin=float(np.mean(pl)), plugin_sd=float(np.std(pl)),
                            mm=float(np.mean(mm)), mm_sd=float(np.std(mm))))
            print(json.dumps(out[-1]), flush=True)
        json.dump(out, open("e3_results.json", "w"), indent=1)

if __name__ == "__main__":
    run()
