"""
Shared helpers for the final-stage experiments (Group 2, HIT401 Capstone, CDU).

Loading and cleaning are identical to ckd_pipeline.py (the interim pipeline),
so every experiment in this repository sees exactly the same data.
"""
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.environ.get("CKD_DATA", os.path.join(ROOT, "data", "kidney_disease.csv"))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results", "final")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

NUM = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
       "hemo", "pcv", "wc", "rc"]
CAT = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
BIN_MAP = {"normal": 0, "abnormal": 1, "notpresent": 0, "present": 1,
           "no": 0, "yes": 1, "good": 0, "poor": 1}
TARGET = "classification"

# Hyper-parameters selected by grid search on the training partition in the
# interim pipeline (results/model_results.csv). Used where an experiment needs
# fixed settings so that only the data condition changes.
STAGE1_PARAMS = {
    "Random Forest": dict(n_estimators=100, max_depth=None, min_samples_leaf=1),
    "SVM (RBF)": dict(C=0.5, gamma=0.05),
    "Decision Tree": dict(criterion="gini", max_depth=3, min_samples_leaf=1),
    "Gradient Boosting": dict(n_estimators=300, learning_rate=0.1, max_depth=2),
}
GRIDS = {
    "Random Forest": {"n_estimators": [100, 300], "max_depth": [None, 6, 10],
                      "min_samples_leaf": [1, 2]},
    "SVM (RBF)": {"C": [0.5, 1, 10], "gamma": ["scale", 0.05]},
    "Decision Tree": {"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5],
                      "criterion": ["gini", "entropy"]},
    "Gradient Boosting": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1],
                          "max_depth": [2, 3]},
}
CLASSICAL = list(STAGE1_PARAMS)


def make_classical(name, params=None):
    p = dict(params or {})
    if name == "Random Forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=1, **p)
    if name == "SVM (RBF)":
        return SVC(probability=True, random_state=SEED, **p)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=SEED, **p)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=SEED, **p)
    raise ValueError(name)


def load_clean(path=DATA):
    """Return encoded features X (NaN kept, original column order) and label y."""
    df = pd.read_csv(path)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
        df[c] = df[c].replace({"nan": np.nan, "": np.nan, "?": np.nan})
    for c in ["pcv", "wc", "rc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = df[TARGET].map({"ckd": 1, "notckd": 0})
    for c in CAT:
        df[c] = df[c].map(BIN_MAP)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=[TARGET]).astype(float)
    y = df[TARGET].astype(int)
    return X, y


def prep(Xtr, Xte, indicators=False, indicator_cols=None):
    """Median (numeric) / mode (binary) imputation and standard scaling,
    fitted on the training rows only. Optionally appends 0/1 missingness
    indicators for the columns in `indicator_cols`."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    if indicators:
        cols = indicator_cols or [c for c in Xtr.columns]
        Mtr = Xtr[cols].isna().astype(float).add_suffix("_missing")
        Mte = Xte[cols].isna().astype(float).add_suffix("_missing")
    num = [c for c in NUM if c in Xtr.columns]
    cat = [c for c in CAT if c in Xtr.columns]
    if num:
        imp = SimpleImputer(strategy="median").fit(Xtr[num])
        Xtr[num], Xte[num] = imp.transform(Xtr[num]), imp.transform(Xte[num])
    if cat:
        imp = SimpleImputer(strategy="most_frequent").fit(Xtr[cat])
        Xtr[cat], Xte[cat] = imp.transform(Xtr[cat]), imp.transform(Xte[cat])
    if indicators:
        Xtr, Xte = pd.concat([Xtr, Mtr], axis=1), pd.concat([Xte, Mte], axis=1)
    sc = StandardScaler().fit(Xtr)
    return (pd.DataFrame(sc.transform(Xtr), columns=Xtr.columns, index=Xtr.index),
            pd.DataFrame(sc.transform(Xte), columns=Xte.columns, index=Xte.index))


def folds(X, y, n_splits=5, n_repeats=5, seed=SEED):
    """5 x 5 repeated stratified folds, identical for every experiment."""
    return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                        random_state=seed).split(X, y))


def metrics(y_true, prob, thr=0.5):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob) if both else np.nan,
        "brier": brier_score_loss(y_true, prob),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
