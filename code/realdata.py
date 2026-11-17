"""Real datasets available offline (sklearn built-ins, ISLP and pydataset bundled CSVs).
All features standardized; categorical columns one-hot encoded; returns X (float32), y (int, 0..K-1)."""
import numpy as np, pandas as pd, os, warnings
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

def _finish(X, y):
    X = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
    _, y = np.unique(np.asarray(y), return_inverse=True)
    return X, y.astype(int)

def _islp(name):
    import ISLP
    return pd.read_csv(os.path.join(os.path.dirname(ISLP.__file__), "data", name + ".csv"))

def _balance(X, y, rng):
    idx = [np.flatnonzero(y == k) for k in np.unique(y)]
    m = min(len(i) for i in idx)
    keep = np.concatenate([rng.choice(i, m, replace=False) for i in idx])
    return X[keep], y[keep]

def load(name, seed=0):
    rng = np.random.default_rng(seed)
    if name == "wdbc":
        d = load_breast_cancer(); return _finish(d.data, d.target)
    if name == "digits":
        d = load_digits(); return _finish(d.data, d.target)
    if name == "oj":
        df = _islp("OJ"); y = df.pop("Purchase")
        df = pd.get_dummies(df, drop_first=True); return _finish(df.values, y)
    if name == "caravan":   # balanced, as in the earlier papers' preprocessing of imbalanced sets
        df = _islp("Caravan"); y = (df.pop("Purchase") == "Yes").astype(int).values
        X, y = _balance(df.values.astype(float), y, rng); return _finish(X, y)
    if name == "biopsy":
        from pydataset import data
        df = data("biopsy").dropna(); y = df["class"]; X = df[[f"V{i}" for i in range(1, 10)]].values
        return _finish(X, y)
    raise ValueError(name)

DATASETS = ["wdbc", "biopsy", "oj", "caravan", "digits"]

if __name__ == "__main__":
    for n in DATASETS:
        X, y = load(n); print(n, X.shape, np.bincount(y))
