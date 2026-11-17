"""Stress tests for the label-free (consensus) selection criterion, with a known Bayes classifier.
Task: X in [0,1]^6, eta(x) = 0.5 + 0.4 tanh(8 (x2 - 0.5 - 0.3 sin(3 pi x1))); x3..x6 are nuisance.
Training/validation labels: Y ~ Bern(eta), optionally followed by class-dependent flips 0->1 w.p. f.
18 candidates: 9 from a 'linear' family (all share the same misspecification bias on the curved boundary)
and 9 diverse flexible learners. Each candidate is trained R times on INDEPENDENT fresh samples.
Stores modal/test predictions so candidate pools can be formed offline."""
import numpy as np, sys, warnings
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

D = 6
def eta(X): return 0.5 + 0.4 * np.tanh(8 * (X[:, 1] - 0.5 - 0.3 * np.sin(3 * np.pi * X[:, 0])))

def sample(n, f, rng):
    X = rng.random((n, D)); y = (rng.random(n) < eta(X)).astype(int)
    flip = (y == 0) & (rng.random(n) < f); y[flip] = 1
    return X, y

LINEAR = {
    "logit C=.01": lambda: LogisticRegression(C=0.01), "logit C=.1": lambda: LogisticRegression(C=0.1),
    "logit C=1": lambda: LogisticRegression(C=1), "logit C=10": lambda: LogisticRegression(C=10),
    "logit C=100": lambda: LogisticRegression(C=100), "linSVC C=.1": lambda: LinearSVC(C=0.1),
    "linSVC C=1": lambda: LinearSVC(C=1), "LDA": lambda: LinearDiscriminantAnalysis(),
    "ridge clf": lambda: RidgeClassifier()}
DIVERSE = {
    "kNN k=15": lambda: KNeighborsClassifier(15), "kNN k=1": lambda: KNeighborsClassifier(1),
    "RF": lambda: RandomForestClassifier(100, min_samples_leaf=3), "ExtraTrees": lambda: ExtraTreesClassifier(100, min_samples_leaf=3),
    "GB": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3), "SVM-RBF": lambda: SVC(C=1, gamma=2.0),
    "MLP64": lambda: MLPClassifier((64, 64), max_iter=600), "tree d=6": lambda: DecisionTreeClassifier(max_depth=6),
    "tree full": lambda: DecisionTreeClassifier()}
CANDS = {**LINEAR, **DIVERSE}

def fit_one(name, mk, n, f, R, Xv, Xt, seed):
    Pv = np.zeros((R, len(Xv)), np.int8); Pt = np.zeros((R, len(Xt)), np.int8)
    for r in range(R):
        rng = np.random.default_rng(seed * 1000 + r)          # same training samples for every candidate
        X, y = sample(n, f, rng); m = mk()
        if hasattr(m, "random_state"): m.set_params(random_state=int(rng.integers(1e9)))
        m.fit(X, y); Pv[r] = m.predict(Xv); Pt[r] = m.predict(Xt)
    return Pv, Pt

def run(f, seed, n=400, R=10, nv=300, M=5000):
    rng = np.random.default_rng(10_000 + seed)
    Xv, yv = sample(nv, f, rng); Xt = rng.random((M, D))
    res = Parallel(n_jobs=2)(delayed(fit_one)(k, mk, n, f, R, Xv, Xt, seed) for k, mk in CANDS.items())
    np.savez_compressed(f"sc_f{f}_s{seed}.npz", names=np.array(list(CANDS)), fam=np.array([0] * len(LINEAR) + [1] * len(DIVERSE)),
                        Pv=np.stack([r[0] for r in res]), Pt=np.stack([r[1] for r in res]), yv=yv, eta_t=eta(Xt), f=f)

if __name__ == "__main__":
    for f in [0.0, 0.1, 0.2, 0.3, 0.4]:
        for s in range(12):   # 12 replicates per condition (Supp. Sec. I.6)
            run(f, s); print("done", f, s, flush=True)
