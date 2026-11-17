"""Two-stage selection on Protocol A (r4 data): screen by mean TRAINING error, then min-V; regret against the best of ALL 19."""
import numpy as np, glob, json
from r4_analyse import mm_entropy_and_pi
rows = []
for f in sorted(glob.glob("r4_*.npz")):
    z = np.load(f); ds, rest = f[3:-4].split("_", 1); kind, sd = rest.rsplit("_s", 1)
    Pv, Pt, yv, yt, K, tr = z["Pv"], z["Pt"], z["yv"], z["yt"], int(z["K"]), z["train_err"]
    test = np.array([np.mean(Pt[l] != yt) for l in range(len(Pt))]); best = test.min()
    V = np.array([mm_entropy_and_pi(Pt[l], K)[0] for l in range(len(Pt))])
    hold = np.array([np.mean(Pv[l] != yv) for l in range(len(Pt))])
    r = dict(file=f, noise=kind, random=float(test.mean() - best), reg_V=float(test[np.argmin(V)] - best),
             reg_holdout=float(test[np.argmin(hold)] - best), reg_train=float(test[np.argmin(tr)] - best))
    for lab, keep in [("abs02", tr < 0.02), ("abs05", tr < 0.05), ("rel01", tr <= tr.min() + 0.01), ("rel02", tr <= tr.min() + 0.02),
                      ("rel05", tr <= tr.min() + 0.05), ("half", tr <= np.median(tr))]:
        if not keep.any(): keep = tr <= tr.min()
        r["reg_scr_" + lab] = float(test[np.argmin(np.where(keep, V, np.inf))] - best); r["nkeep_" + lab] = int(keep.sum())
    rows.append(r)
json.dump(rows, open("r8_protA.json", "w"), indent=1)
keys = [k for k in rows[0] if k.startswith("reg_")]
for g, lab in [("sym", "symmetric"), ("classdep", "class-dep"), ("", "all")]:
    R = [r for r in rows if g in r["noise"]]
    print(f"{lab:10s} n={len(R)} random={np.mean([r['random'] for r in R]):.4f} " + " ".join(f"{k[4:]}={np.mean([r[k] for r in R]):.4f}" for k in keys))
print("mean kept:", {k: np.mean([r[k] for r in rows]) for k in rows[0] if k.startswith("nkeep")})
