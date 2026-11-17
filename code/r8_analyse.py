"""Analysis of R8 (Protocol B): (i) two-stage selection = screen by TRAINING fit, then min-V2 (or the consensus criterion);
(ii) CMI-based selection (evaluated/loss CMI and functional CMI on the supersample pairs), at matched and larger cost.
Regret target and the reference criteria (holdout, 2-/5-fold CV, consensus criterion) come from the Protocol B files r6_*.npz;
r8 draw 0 reproduces their 2-fold models exactly. Pools: all 19 configurations, and the 6 overparameterized ones.
Training-fit screens use only the training labels (no held-out labels). CMI estimates use the labels of both members of each
pair (that is how CMI is defined: it conditions on the supersample, labels included), so CMI criteria are NOT label-free."""
import numpy as np, glob, json, re
from collections import defaultdict
from r6_analyse import vstats, labelfree, plurality
OVER = ["kNN k=1", "Tree depth=None", "RF leaf=1", "GB 300x d6 lr1", "SVM-RBF C=10000", "MLP512 alpha=1e-05"]
LN2 = np.log(2)

def H(counts):
    p = counts[counts > 0] / counts.sum(); return float(-(p * np.log(p)).sum())

def pair_mi(pat, u):
    """pat: (n_obs, pairs) integer pattern codes, u: (n_obs, pairs) selection bits -> per-pair plug-in I(pattern; U) in nats."""
    n_obs, P = pat.shape; out = np.zeros(P)
    for i in range(P):
        a, b = pat[:, i], u[:, i]
        _, ja = np.unique(a, return_inverse=True); joint = np.zeros((ja.max() + 1, 2)); np.add.at(joint, (ja, b), 1)
        out[i] = H(joint.sum(1)) + H(joint.sum(0)) - H(joint.ravel())
    return out

def legacy(f, y):
    j = f.astype(int) * (y.max() + 1) + y; return (H(np.bincount(f)) + H(np.bincount(y)) - H(np.bincount(j))) / H(np.bincount(y))

def analyse_case(f8):
    z8 = np.load(f8); f6 = f8.replace("r8_", "r6_"); z6 = np.load(f6)
    names = list(z8["names"]); assert names == list(z6["names"])
    K = int(z8["K"]); yn = z8["ytr_noisy"]; yte = z6["yte"]; ptr, pte, U, pairs = z8["ptr"], z8["pte"], z8["U"], z8["pairs"]
    L, D = ptr.shape[0], ptr.shape[1]; f2 = z8["fold2"]
    assert (pte[:, 0] == z6["testfold2"]).all(), "draw 0 does not reproduce Protocol B"
    err = (z6["test_refit"] != yte).mean(1)
    # training fit of the draw-0 fold models on their own halves (model 0 trained on fold 1, model 1 on fold 0)
    own = [f2 == 1, f2 == 0]
    tr = np.array([np.mean([(ptr[l, 0, s][own[s]] != yn[own[s]]).mean() for s in (0, 1)]) for l in range(L)])
    leg = np.array([np.mean([legacy(ptr[l, 0, s][own[s]], yn[own[s]]) for s in (0, 1)]) for l in range(L)])
    cv2 = (z6["oof2"] != yn).mean(1)                                   # pooled out-of-fold error, as in r6_analyse.py
    assert all((ptr[l, 0, s][~own[s]] == z6["oof2"][l][~own[s]]).all() for l in range(L) for s in (0, 1))
    # repeated 2-fold CV over the D draws (same fits as CMI with D draws)
    m = pairs.shape[0]; ar = np.arange(m); trained = np.zeros((D, 2, len(yn)), bool)
    for d in range(D):
        if d == 0: trained[0, 0], trained[0, 1] = f2 == 1, f2 == 0
        else:
            A = np.concatenate([pairs[ar, U[d]], z8["left"]]); B = pairs[ar, 1 - U[d]]
            trained[d, 0, A] = True; trained[d, 1, B] = True
    cvD = np.array([np.mean([(ptr[l, d, s][~trained[d, s]] != yn[~trained[d, s]]).mean() for d in range(D) for s in (0, 1)]) for l in range(L)])
    trD = np.array([np.mean([(ptr[l, d, s][trained[d, s]] != yn[trained[d, s]]).mean() for d in range(D) for s in (0, 1)]) for l in range(L)])
    # V2 per draw (each draw is a disjoint pair) and averaged
    V2 = vstats(pte[:, 0], K)[0]; V2D = np.mean([vstats(pte[:, d], K)[0] for d in range(D)], 0)
    _, pib, modes = vstats(pte[:, 0], K)
    # CMI: observations = (draw, side); selection bit of pair i for model (d,s) = index of the trained member
    def cmis(nd):
        obs_u = np.array([U[d] if s == 0 else 1 - U[d] for d in range(nd) for s in (0, 1)])
        e, fc = np.zeros(L), np.zeros(L); ei = np.zeros(L)
        for l in range(L):
            P = np.array([ptr[l, d, s][pairs] for d in range(nd) for s in (0, 1)])          # (obs, pairs, 2) predictions
            Ls = (P != yn[pairs][None]).astype(int)                                         # losses on both members
            mi_e = pair_mi(Ls[..., 0] * 2 + Ls[..., 1], obs_u); mi_f = pair_mi(P[..., 0] * K + P[..., 1], obs_u)
            e[l], fc[l] = mi_e.mean(), mi_f.mean(); ei[l] = np.sqrt(2 * mi_e).mean()
        return e, fc, ei
    e1, f1, ei1 = cmis(1); eD, fD, eiD = cmis(D)
    hold = (z6["hold_pred"] != yn[z6["hold_idx"]]).mean(1); cv5 = (z6["oof5"] != yn).mean(1)
    # label-free noise-rate estimate: consensus of all 19 configurations' out-of-fold predictions vs the noisy labels
    rho = float((plurality(z6["oof2"], K) != yn).mean())
    crit = dict(holdout1=hold, cv2=cv2, cv5=cv5, cv2x3=cvD, V2=V2, V2x3=V2D, train=tr,
                ecmi1=e1, fcmi1=f1, ecmi1_bound=tr + np.sqrt(2 * e1), ecmi1_ind=tr + ei1,
                ecmi3=eD, fcmi3=fD, ecmi3_bound=trD + np.sqrt(2 * eD), fcmi3_bound=trD + np.sqrt(2 * fD), ecmi3_ind=trD + eiD)
    t8 = z8["times"]; t6 = z6["times"]
    out = dict(file=f6, rho_hat=rho, rho=float((yn != z8["ytr_clean"]).mean()))
    for pool, idx in (("all", np.arange(L)), ("op", np.array([names.index(n) for n in OVER]))):
        e = err[idx]; best = e.min(); reg = lambda c: float(e[np.argmin(c[idx] if len(c) == L else c)] - best)
        r = {k: reg(c) for k, c in crit.items()}
        lf = labelfree(modes, pib, K)                               # consensus over all 19 (as in the paper)
        r["lf2"] = reg(lf)
        screens = {"interp": tr <= 0.01, "rel02": tr <= tr[idx].min() + 0.02, "rho02": tr <= rho + 0.02, "half": tr <= np.median(tr[idx]),
                   "legacy90": leg >= 0.90}
        for sname, keep in screens.items():
            k = keep[idx]
            if not k.any(): k = tr[idx] <= tr[idx].min()
            r[f"scr_{sname}_V2"] = float(e[np.argmin(np.where(k, V2[idx], np.inf))] - best)
            r[f"scr_{sname}_lf2"] = float(e[np.argmin(np.where(k, lf[idx], np.inf))] - best)
            r[f"nkeep_{sname}"] = int(k.sum()); r[f"bestkept_{sname}"] = bool(k[np.argmin(e)])
        r["random"] = float(e.mean() - best)
        c6 = t6[idx].sum(0); c8 = t8[idx].sum(0)                      # r6 cols: refit, holdout, cv2, cv5, unlab2, unlab5
        h1 = c6[1]
        r["cost"] = dict(holdout1=1.0, cv2=c6[2] / h1, cv5=c6[3] / h1, V2=(c6[2] + c6[4]) / h1, cmi1=c8[0].sum() / h1,
                         cmi3=c8.sum() / h1, cv2x3=c8.sum() / h1, V2x3=(c8.sum() + D * c6[4]) / h1)
        out[pool] = r
    return out

def key(f):
    m = re.match(r"r6_(.+)_(sym0\.2|classdep|instdep)_s(\d+)\.npz", f); return m.group(1), m.group(2), int(m.group(3))

def cboot(rows, pool, a, b, B=4000, seed=0):
    C = defaultdict(list)
    for r in rows: ds, nz, s = key(r["file"]); C[(ds, nz)].append(r[pool][a] - r[pool][b])
    C = [np.array(v) for v in C.values()]; g = np.random.default_rng(seed); pt = np.concatenate(C).mean()
    bs = [np.concatenate([C[i] for i in g.integers(0, len(C), len(C))]).mean() for _ in range(B)]
    return float(pt), [float(x) for x in np.percentile(bs, [2.5, 97.5])]

def _rows_to_tex(S, rows, cols, cost_pool, ex=None, extra_all=False):
    keys = [r[1] for r in rows if r[1]]; best = {c: min(S[c[0]][k][c[1]] for k in keys) for c in cols}; out = []
    bestA = min(S["all"][k]["all"] for k in keys); bestE = max(ex(k) for k in keys) if ex else None
    for lab, k, lbl, c in rows:
        if k is None: out.append(lab + "\\\\"); continue
        v = [(f"\\best{{{S[a][k][g]:.4f}}}" if abs(S[a][k][g] - best[(a, g)]) < 1e-12 else f"{S[a][k][g]:.4f}") for a, g in cols]
        cells = ([lbl] if lbl else []) + v
        if ex: e = ex(k); cells.append(f"\\best{{{e}}}" if e == bestE else str(e))
        cells.append(f"{S[cost_pool]['_cost'][c]:.2f}")
        if extra_all: av = S["all"][k]["all"]; cells.append(f"\\best{{{av:.4f}}}" if abs(av - bestA) < 1e-12 else f"{av:.4f}")
        out.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    return out

def write_table(S, rows_data=None, path="r8_table.tex", path_cmi="r8_cmi_main_table.tex"):
    """Supp. Table S6 (two-stage selection) and main Table 5 (CMI vs min-V), as LaTeX rows."""
    two = [("\\multicolumn{7}{l}{\\emph{Held-out labels}}", None, None, None),
           ("Holdout (80/20)", "holdout1", None, "holdout1"), ("2-fold CV", "cv2", None, "cv2"), ("5-fold CV", "cv5", None, "cv5"),
           ("\\multicolumn{7}{l}{\\emph{No held-out labels, one stage}}", None, None, None),
           ("Min-$V_2$\\prop{}", "V2", None, "V2"), ("Consensus criterion\\prop{}$^\\ast$, 2 halves", "lf2", None, "V2"),
           ("\\multicolumn{7}{l}{\\emph{Two-stage (training labels only): screen by training fit, then min-$V_2$}}", None, None, None),
           ("Keep interpolators (training error $\\le1\\%$)", "scr_interp_V2", None, "V2"), ("Keep legacy LMC score $\\ge0.9$", "scr_legacy90_V2", None, "V2"),
           ("Keep training error below the median", "scr_half_V2", None, "V2"), ("Keep training error $\\le\\hat\\rho+0.02$", "scr_rho02_V2", None, "V2"),
           ("Keep training error $\\le\\hat\\rho+0.02$, then consensus\\prop{}$^\\ast$", "scr_rho02_lf2", None, "V2")]
    out = _rows_to_tex(S, two, [("all", "sym"), ("all", "class"), ("all", "inst"), ("all", "all"), ("op", "all")], "all")
    out.append("\\midrule\n\\emph{Random choice} & " + " & ".join(f"\\emph{{{S[a]['random'][g]:.4f}}}" for a, g in
               [("all", "sym"), ("all", "class"), ("all", "inst"), ("all", "all"), ("op", "all")]) + " & \\\\")
    open(path, "w").write("\n".join(out))
    if rows_data is None: return
    cmi = [("\\multicolumn{8}{l}{\\emph{One pair of half-sample fits}}", None, None, None),
           ("2-fold CV", "cv2", "yes", "cv2"), ("e-CMI", "ecmi1", "yes", "cv2"), ("Training error $+\\sqrt{2\\,\\overline{\\text{e-CMI}}}$", "ecmi1_bound", "yes", "cv2"),
           ("Training error $+$ samplewise e-CMI bound", "ecmi1_ind", "yes", "cv2"), ("Min-$V_2$\\prop{}", "V2", "no", "V2"),
           ("\\multicolumn{8}{l}{\\emph{Three pairs of half-sample fits}}", None, None, None),
           ("2-fold CV, 3 random splits", "cv2x3", "yes", "cv2x3"), ("e-CMI", "ecmi3", "yes", "cv2x3"), ("Training error $+$ samplewise e-CMI bound", "ecmi3_ind", "yes", "cv2x3"),
           ("Min-$V_2$\\prop{}, averaged over the 3 pairs", "V2x3", "no", "V2x3"),
           ("\\multicolumn{8}{l}{\\emph{Reference}}", None, None, None), ("5-fold CV", "cv5", "yes", "cv5")]
    ex = lambda k: sum(r["op"][k] < 1e-12 for r in rows_data)
    out = _rows_to_tex(S, cmi, [("op", "sym"), ("op", "class"), ("op", "inst"), ("op", "all")], "op", ex=ex, extra_all=True)
    open(path_cmi, "w").write("\n".join(out))

if __name__ == "__main__":
    rows = [analyse_case(f) for f in sorted(glob.glob("r8_*_s[0-9].npz"))]
    print(len(rows), "cases")
    G = [("sym0.2", "sym"), ("classdep", "class"), ("instdep", "inst"), ("", "all")]
    summ = {}
    for pool in ("all", "op"):
        keys = [k for k in rows[0][pool] if k not in ("cost",) and not k.startswith(("nkeep", "bestkept"))]
        summ[pool] = {k: {lab: float(np.mean([r[pool][k] for r in rows if g in r["file"]])) for g, lab in G} for k in keys}
        summ[pool]["_cost"] = {c: float(np.median([r[pool]["cost"][c] for r in rows])) for c in rows[0][pool]["cost"]}
        summ[pool]["_nkeep"] = {k: float(np.mean([r[pool][k] for r in rows])) for k in rows[0][pool] if k.startswith(("nkeep", "bestkept"))}
        print(f"\n=== pool {pool}")
        for k in keys: print(f"  {k:22s} " + " ".join(f"{summ[pool][k][lab]:.4f}" for _, lab in G))
        print("  cost", {c: round(v, 2) for c, v in summ[pool]["_cost"].items()})
        print("  kept", {c: round(v, 2) for c, v in summ[pool]["_nkeep"].items()})
    diffs = {}
    for pool, pairs_ in (("all", [("scr_rho02_V2", "V2"), ("scr_rho02_V2", "cv5"), ("scr_rho02_V2", "cv2"), ("scr_rho02_V2", "holdout1"),
                                   ("scr_rho02_lf2", "lf2"), ("scr_rho02_lf2", "cv5"), ("scr_interp_V2", "V2"), ("ecmi1", "cv2"), ("ecmi1", "V2"),
                                   ("ecmi3", "cv2x3"), ("fcmi3", "cv2x3"), ("V2", "ecmi1")]),
                         ("op", [("scr_rho02_V2", "V2"), ("ecmi1", "cv2"), ("ecmi1", "V2"), ("ecmi3", "cv2x3"), ("V2", "ecmi1"), ("V2", "ecmi3"),
                                  ("fcmi3", "V2"), ("ecmi3_bound", "V2")])):
        for a, b in pairs_:
            diffs[f"{pool}:{a}-{b}"] = cboot(rows, pool, a, b); print(f"  {pool}: {a} - {b} = {diffs[f'{pool}:{a}-{b}']}")
    print("noise-rate estimate: mean |rho_hat - rho| =", np.mean([abs(r["rho_hat"] - r["rho"]) for r in rows]))
    json.dump(dict(rows=rows, summary=summ, diffs=diffs), open("r8_analysis.json", "w"), indent=1)
    write_table(summ, rows)
