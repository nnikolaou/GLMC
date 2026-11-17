"""Adaptive clustering: use one-vote-per-cluster only when a single cluster holds >= half of the candidates."""
import numpy as np, glob
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from s_analyse import load_case
from r5_analyse import stats
from r5_remedy import plurality

def labels(mt, tau=0.05):
    A = (mt[:, None, :] == mt[None, :, :]).mean(2); D = 1 - A; np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), "average"), t=tau, criterion="distance")

def adaptive_cons(mt, K):
    lab = labels(mt); u, cnt = np.unique(lab, return_counts=True)
    if cnt.max() >= len(mt) / 2:
        votes = np.stack([plurality(mt[lab == c], K) for c in u]); return plurality(votes, K), True
    return plurality(mt, K), False

# synthetic S1 (f = 0)
rng = np.random.default_rng(0); out = {}
for f in sorted(glob.glob("sc_f0.0_s*.npz")):
    c = load_case(f); lin = np.flatnonzero(c["fam"] == 0); div = np.flatnonzero(c["fam"] == 1)
    for k in (0, 3, 6, 8, 9):
        for _ in range(30 if k else 1):
            pool = np.concatenate([rng.choice(lin, k, replace=False), rng.choice(div, 12 - k, replace=False)]) if k else div
            mt = c["mode"][pool]; cons, used = adaptive_cons(mt, 2)
            D = (mt != cons).mean(1); lf = D + c["pibar"][pool] * np.clip(1 - 2 * D, 0, 1)
            exc = c["exc"][pool]; out.setdefault(k, []).append((exc[lf.argmin()] - exc.min(), used))
for k, v in out.items(): print(f"S1 linear share {k}/12: adaptive label-free regret {np.mean([a for a, _ in v]):.4f}, clustering used in {np.mean([u for _, u in v]):.0%}")
# real data R5
res = []
for f in sorted(glob.glob("r5_*.npz")):
    z = np.load(f); K = int(z["K"]); Pv, Pt, yt = z["Pv"], z["Pt"], z["yt"]; test = (Pt != yt).mean((1, 2))
    V, pibar, mt, mv = stats(Pv, Pt, K); cons, used = adaptive_cons(mt, K)
    D = (mt != cons).mean(1); lf = D + pibar * np.clip(1 - D * K / (K - 1), 0, 1)
    res.append((test[lf.argmin()] - test.min(), used, "classdep" in f))
print("R5 adaptive label-free: symmetric %.4f class-dep %.4f; clustering used in %d/90" % (
    np.mean([a for a, u, c in res if not c]), np.mean([a for a, u, c in res if c]), sum(u for _, u, _ in res)))
