"""Protocol B over 5 seeds per dataset (15 datasets x 3 noise models x 5 seeds = 225 cases).
Reads r6_overparam.json (6 overparameterized candidates) and r6_analysis.json (all 19 candidates).
Means are over all cases; 95% intervals for paired differences use a cluster bootstrap that resamples
(dataset, noise) clusters (45 clusters of 5 seeds), so seeds of the same dataset are not treated as independent."""
import json, re, numpy as np
from collections import defaultdict
def key(f):  # r6_<ds>_<noise>_s<seed>.npz
    m = re.match(r"r6_(.+)_(sym0\.2|classdep|instdep)_s(\d+)\.npz", f); return m.group(1), m.group(2), int(m.group(3))
def clusters(rows):
    C = defaultdict(list)
    for r in rows: ds, nz, s = key(r["file"]); C[(ds, nz)].append(r)
    return C
def cboot(rows, fn, B=4000, seed=0):
    C = list(clusters(rows).values()); g = np.random.default_rng(seed)
    point = fn([r for c in C for r in c]); bs = []
    for _ in range(B):
        pick = g.integers(0, len(C), len(C)); bs.append(fn([r for i in pick for r in C[i]]))
    return point, np.percentile(bs, [2.5, 97.5])
mean = lambda k: (lambda R: np.mean([r["reg_" + k] for r in R]))
diff = lambda a, b: (lambda R: np.mean([r["reg_" + a] - r["reg_" + b] for r in R]))
GROUPS = [("sym0.2", "symmetric"), ("classdep", "class-dependent"), ("instdep", "instance-dependent"), ("", "all")]
out = {}
# ---- overparameterized candidates
op = json.load(open("r6_overparam.json"))
cost_op = {"holdout1": lambda x: 1.0, "cv2": lambda x: x["cost_cv2"] / x["cost_holdout1"], "cv5": lambda x: x["cost_cv5"] / x["cost_holdout1"],
           "cv5_bc": lambda x: x["cost_poolfit5"] / x["cost_holdout1"], "V2": lambda x: (x["cost_cv2"] + x["cost_unlab2"]) / x["cost_holdout1"],
           "V5": lambda x: (x["cost_cv5"] + x["cost_unlab5"]) / x["cost_holdout1"], "lf2": lambda x: x["cost_pool2"] / x["cost_holdout1"],
           "lf5": lambda x: x["cost_pool5"] / x["cost_holdout1"]}
K_OP = list(cost_op)
print(f"OVERPARAMETERIZED SET: {len(op)} cases, {len(clusters(op))} clusters; seeds {sorted(set(key(r['file'])[2] for r in op))}")
tab = {}
for k in K_OP:
    tab[k] = {lab: float(np.mean([r["reg_" + k] for r in op if g in r["file"]])) for g, lab in GROUPS}
    tab[k]["cost"] = float(np.median([cost_op[k](r) for r in op])); tab[k]["exact_frac"] = float(np.mean([r["reg_" + k] < 1e-12 for r in op]))
    print(f"  {k:9s} " + " ".join(f"{tab[k][lab]:.4f}" for _, lab in GROUPS) + f"  cost {tab[k]['cost']:.2f}  exact {tab[k]['exact_frac']:.2f}")
tab["random"] = {lab: float(np.mean([r["random"] for r in op if g in r["file"]])) for g, lab in GROUPS}
print("  random   ", " ".join(f"{tab['random'][lab]:.4f}" for _, lab in GROUPS))
out["overparam"] = tab; out["overparam_diffs"] = {}
for a, b in [("V2", "holdout1"), ("V2", "cv2"), ("V5", "cv5"), ("V2", "V5"), ("lf2", "holdout1"), ("lf2", "cv2"), ("lf2", "cv5"), ("lf2", "V2"),
             ("cv5", "holdout1"), ("cv2", "holdout1")]:
    out["overparam_diffs"][f"{a}-{b}"] = {}
    for g, lab in GROUPS:
        R = [r for r in op if g in r["file"]]; p, (lo, hi) = cboot(R, diff(a, b))
        out["overparam_diffs"][f"{a}-{b}"][lab] = [float(p), float(lo), float(hi)]
    print(f"  {a}-{b}: " + "; ".join(f"{lab} {v[0]:+.4f}[{v[1]:+.4f},{v[2]:+.4f}]" for lab, v in out["overparam_diffs"][f"{a}-{b}"].items()))
# seed 0 vs seeds 1-4 (replication check)
for k in ["holdout1", "cv2", "cv5", "V2", "lf2"]:
    s0 = np.mean([r["reg_" + k] for r in op if key(r["file"])[2] == 0]); s14 = np.mean([r["reg_" + k] for r in op if key(r["file"])[2] > 0])
    out.setdefault("overparam_seed0_vs_rest", {})[k] = [float(s0), float(s14)]
print("  seed 0 vs seeds 1-4:", {k: [round(a, 4), round(b, 4)] for k, (a, b) in out["overparam_seed0_vs_rest"].items()})
# ---- all 19 candidates
al = json.load(open("r6_analysis.json"))
from r6_analyse import CRIT, cost
K_ALL = ["holdout1", "cv2", "cv5", "cv5_bc", "V2", "V5", "lf2", "lf5", "lfa5", "switch2", "switch5"]
spec = {k: s for k, _, s in CRIT}; spec.update({"V2": "cv2+unlab2", "switch2": "cv2+unlab2"})
print(f"\nALL 19 CANDIDATES: {len(al)} cases")
ta = {}
for k in K_ALL:
    ta[k] = {lab: float(np.mean([r["reg_" + k] for r in al if g in r["file"]])) for g, lab in GROUPS}
    ta[k]["cost"] = float(np.median([cost(r, spec[k]) / r["cost_holdout1"] for r in al]))
    print(f"  {k:9s} " + " ".join(f"{ta[k][lab]:.4f}" for _, lab in GROUPS) + f"  cost {ta[k]['cost']:.2f}")
ta["random"] = {lab: float(np.mean([r["random"] for r in al if g in r["file"]])) for g, lab in GROUPS}
out["all"] = ta; out["all_diffs"] = {}
for a, b in [("cv5_bc", "cv5"), ("lf5", "cv5"), ("lf2", "cv2"), ("V5", "cv5"), ("switch5", "cv5"), ("lf2", "holdout1"), ("cv5", "holdout1")]:
    out["all_diffs"][f"{a}-{b}"] = {}
    for g, lab in GROUPS:
        R = [r for r in al if g in r["file"]]; p, (lo, hi) = cboot(R, diff(a, b))
        out["all_diffs"][f"{a}-{b}"][lab] = [float(p), float(lo), float(hi)]
    print(f"  {a}-{b}: " + "; ".join(f"{lab} {v[0]:+.4f}[{v[1]:+.4f},{v[2]:+.4f}]" for lab, v in out["all_diffs"][f"{a}-{b}"].items()))
out["switch_trigger_frac"] = {lab: float(np.mean([r["asym5"] > 0.1 for r in al if g in r["file"]])) for g, lab in GROUPS}
print("  switch triggers:", out["switch_trigger_frac"])
json.dump(out, open("r6_multiseed.json", "w"), indent=1)
