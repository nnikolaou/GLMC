"""Width selection among interpolating (minimum-norm, p >= n) random-features models (main Sec. 7.2, Supp. Table S2).
Reads e2_results.json (synthetic) and r3_results.json (real); prints Spearman(V, error), min-V regret, random regret."""
import json, numpy as np
from scipy.stats import spearmanr
def report(rows, key_fields, err_key):
    for k in sorted(set(tuple(r[f] for f in key_fields) for r in rows)):
        rs = [r for r in rows if tuple(r[f] for f in key_fields) == k and r["p"] >= r["n"]]
        if len(rs) < 3: continue
        e = np.array([r[err_key] for r in rs]); V = np.array([r["V"] for r in rs])
        print(k, f"n_widths={len(rs)} spearman={spearmanr(V, e)[0]:.3f} minV_regret={e[V.argmin()] - e.min():.4f} random_regret={e.mean() - e.min():.4f}")
report([r for r in json.load(open("e2_results.json")) if r["lam"] == 0], ["eps", "lam"], "excess_risk")
report([r for r in json.load(open("r3_results.json")) if r["lam"] == 0], ["dataset", "rho", "lam"], "test_err")
