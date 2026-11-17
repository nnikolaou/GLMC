"""Offline analysis of R4: selection regret of holdout, V, and combinations, for full and small validation sets."""
import numpy as np, glob, json, sys
from realdata import DATASETS

def mm_entropy_and_pi(P, K):
    R, M = P.shape
    counts = np.stack([(P == k).sum(0) for k in range(K)])
    q = counts / R
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(np.where(q > 0, q * np.log2(q), 0)).sum(0)
    H = H + ((counts > 0).sum(0) - 1) / (2 * R * np.log(2))
    return H.mean(), (1 - q.max(0)).mean(), counts.argmax(0)

def criteria(Pv, Pt, yv, K, cols):
    """Pv: (L,R,mv) val preds; Pt: (L,R,M) test preds; cols: indices of the validation points used."""
    out = {}
    L = Pv.shape[0]; m = len(cols)
    hold = np.array([np.mean(Pv[l][:, cols] != yv[cols]) for l in range(L)])
    V, pibar, modal_v = np.zeros(L), np.zeros(L), np.zeros(L)
    for l in range(L):
        V[l], pibar[l], _ = mm_entropy_and_pi(Pt[l], K)
        _, _, mode = mm_entropy_and_pi(Pv[l][:, cols], K)
        modal_v[l] = np.mean(mode != yv[cols])
    comb = modal_v + pibar * np.clip(1 - modal_v * K / (K - 1), 0, 1)   # decomposition: modal error + expression term
    best = hold.min(); se = np.sqrt(best * (1 - best) / m)
    gate = np.where(hold <= best + se, V, np.inf)                       # one-SE rule on holdout, then least V
    out.update(holdout=hold, V=V, modal_holdout=modal_v, combined=comb, gated=gate)
    return out

def analyse(m_small=40, draws=100, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for f in sorted(glob.glob("r4_*.npz")):
        z = np.load(f); ds, rest = f[3:-4].split("_", 1); kind, sd = rest.rsplit("_s", 1)
        Pv, Pt, yv, yt, K = z["Pv"], z["Pt"], z["yv"], z["yt"], int(z["K"])
        test = np.array([np.mean(Pt[l] != yt) for l in range(len(Pt))])
        interp = z["train_err"] < 0.02
        for mlab, colsets in [("full", [np.arange(len(yv))]),
                              (f"m={m_small}", [rng.choice(len(yv), m_small, replace=False) for _ in range(draws)])]:
            reg = {k: [] for k in ["holdout", "V", "modal_holdout", "combined", "gated"]}; regI = {k: [] for k in reg}
            for cols in colsets:
                c = criteria(Pv, Pt, yv, K, cols)
                for k in reg:
                    reg[k].append(test[np.argmin(c[k])] - test.min())
                    if interp.sum() > 3:
                        if k == "gated":   # re-gate within the interpolators
                            hb = c["holdout"][interp].min(); se = np.sqrt(hb * (1 - hb) / len(cols))
                            ci = np.where(interp & (c["holdout"] <= hb + se), c["V"], np.inf)
                        else:
                            ci = np.where(interp, c[k], np.inf)
                        regI[k].append(test[np.argmin(ci)] - test[interp].min())
            row = dict(dataset=ds, noise=kind, split=int(sd), val=mlab, n_val=len(yv), random=float(test.mean() - test.min()),
                       **{f"reg_{k}": float(np.mean(v)) for k, v in reg.items()})
            if interp.sum() > 3: row.update({f"regI_{k}": float(np.mean(v)) for k, v in regI.items()}, n_interp=int(interp.sum()))
            rows.append(row)
    json.dump(rows, open("r4_analysis.json", "w"), indent=1)
    return rows

def summarize(rows):
    keys = ["holdout", "V", "modal_holdout", "combined", "gated"]
    for noise_set, lab in [(("sym0.2", "sym0.3"), "symmetric"), (("classdep",), "class-dependent")]:
        for val in ["full", "m=40"]:
            R = [r for r in rows if r["noise"] in noise_set and r["val"] == val]
            if not R: continue
            s = " ".join(f"{k}={np.mean([r['reg_'+k] for r in R]):.4f}" for k in keys)
            RI = [r for r in R if "regI_V" in r]
            si = " ".join(f"{k}={np.mean([r['regI_'+k] for r in RI]):.4f}" for k in keys) if RI else ""
            print(f"{lab:16s} {val:5s} n={len(R)} random={np.mean([r['random'] for r in R]):.3f} | ALL {s} | INTERP(n={len(RI)}) {si}")

def boot_diff(rows, a, b, noise_set, val, B=5000, seed=0, prefix="reg_"):
    R = [r for r in rows if r["noise"] in noise_set and r["val"] == val and prefix + a in r]
    d = np.array([r[prefix + a] - r[prefix + b] for r in R]); rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(B)])
    return d.mean(), np.percentile(bs, [2.5, 97.5]), len(d)

if __name__ == "__main__":
    rows = analyse(); summarize(rows)
