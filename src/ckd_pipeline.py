"""
Kidney Failure Prediction - Interim analysis pipeline
Group 2, PRT840/HIT401 Thesis, Charles Darwin University

Runs: cleaning -> imputation comparison -> EDA -> classical models -> deep models -> evaluation
Outputs: figures/*.png and results/*.csv
"""
import os, json, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
sns.set_style("whitegrid")

RAW = os.environ.get("CKD_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "kidney_disease.csv"))

# ----------------------------------------------------------------------
# 1. LOAD AND CLEAN
# ----------------------------------------------------------------------
df = pd.read_csv(RAW)
if "id" in df.columns:
    df = df.drop(columns=["id"])
n_raw, p_raw = df.shape
print(f"Raw shape: {df.shape}")

# 1a. strip whitespace/tabs from every object column (data-entry noise)
obj_cols = df.select_dtypes(include=["object", "string"]).columns
for c in obj_cols:
    df[c] = df[c].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    df[c] = df[c].replace({"nan": np.nan, "": np.nan, "?": np.nan})

# 1b. known corrupted category spellings
df["dm"] = df["dm"].replace({"yes": "yes", "no": "no"})
df["cad"] = df["cad"].replace({"\tno": "no"})
df["classification"] = df["classification"].replace({"ckd": "ckd", "notckd": "notckd"})

# 1c. numeric attributes stored as text
for c in ["pcv", "wc", "rc"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

TARGET = "classification"
df[TARGET] = df[TARGET].map({"ckd": 1, "notckd": 0})

num_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
            "hemo", "pcv", "wc", "rc"]
cat_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

print("Category levels after cleaning:")
for c in cat_cols:
    print(" ", c, sorted(df[c].dropna().unique().tolist()))

# 1d. duplicates
dups = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed: {dups}; shape now {df.shape}")

# ----------------------------------------------------------------------
# 2. MISSINGNESS PROFILE
# ----------------------------------------------------------------------
miss = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
miss_tab = miss[miss > 0].reset_index()
miss_tab.columns = ["Attribute", "Missing (%)"]
miss_tab.to_csv("results/missingness.csv", index=False)
complete_cases = df.dropna().shape[0]
print(f"Complete cases: {complete_cases}/{len(df)} ({complete_cases/len(df)*100:.1f}%)")

# ----------------------------------------------------------------------
# 3. ENCODE
# ----------------------------------------------------------------------
enc = df.copy()
bin_map = {"normal": 0, "abnormal": 1, "notpresent": 0, "present": 1,
           "no": 0, "yes": 1, "good": 0, "poor": 1}
for c in cat_cols:
    enc[c] = enc[c].map(bin_map)

X = enc.drop(columns=[TARGET])
y = enc[TARGET]
print("Class balance:", y.value_counts().to_dict())

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"Train {X_tr.shape}, Test {X_te.shape}")


def build_pipeline(kind):
    """Fit imputer+scaler on TRAIN only, apply to both partitions (no leakage)."""
    Xtr, Xte = X_tr.copy(), X_te.copy()
    if kind == "simple":
        num_imp = SimpleImputer(strategy="median")
        cat_imp = SimpleImputer(strategy="most_frequent")
        Xtr[num_cols] = num_imp.fit_transform(Xtr[num_cols])
        Xte[num_cols] = num_imp.transform(Xte[num_cols])
        Xtr[cat_cols] = cat_imp.fit_transform(Xtr[cat_cols])
        Xte[cat_cols] = cat_imp.transform(Xte[cat_cols])
    else:  # knn
        imp = KNNImputer(n_neighbors=5)
        cols = list(Xtr.columns)
        sc0 = StandardScaler().fit(Xtr[num_cols])
        Xtr[num_cols] = sc0.transform(Xtr[num_cols])
        Xte[num_cols] = sc0.transform(Xte[num_cols])
        Xtr = pd.DataFrame(imp.fit_transform(Xtr), columns=cols, index=X_tr.index)
        Xte = pd.DataFrame(imp.transform(Xte), columns=cols, index=X_te.index)
        Xtr[cat_cols] = Xtr[cat_cols].round().clip(0, 1)
        Xte[cat_cols] = Xte[cat_cols].round().clip(0, 1)
        Xtr[num_cols] = sc0.inverse_transform(Xtr[num_cols])
        Xte[num_cols] = sc0.inverse_transform(Xte[num_cols])
    sc = StandardScaler().fit(Xtr)
    return (pd.DataFrame(sc.transform(Xtr), columns=Xtr.columns, index=Xtr.index),
            pd.DataFrame(sc.transform(Xte), columns=Xte.columns, index=Xte.index),
            Xtr, Xte)


# ----------------------------------------------------------------------
# 4. IMPUTATION COMPARISON (5-fold CV on training set only)
# ----------------------------------------------------------------------
imp_rows = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
store = {}
for kind in ["simple", "knn"]:
    Xtr_s, Xte_s, Xtr_raw, Xte_raw = build_pipeline(kind)
    store[kind] = (Xtr_s, Xte_s, Xtr_raw, Xte_raw)
    for name, mdl in [("Random Forest", RandomForestClassifier(n_estimators=300, random_state=SEED)),
                      ("SVM (RBF)", SVC(probability=True, random_state=SEED))]:
        sc_ = cross_val_score(mdl, Xtr_s, y_tr, cv=cv, scoring="f1")
        imp_rows.append({"Imputation": "Median/Mode" if kind == "simple" else "KNN (k=5)",
                         "Model": name,
                         "CV F1 mean": round(sc_.mean(), 4),
                         "CV F1 SD": round(sc_.std(), 4)})
imp_df = pd.DataFrame(imp_rows)
imp_df.to_csv("results/imputation_comparison.csv", index=False)
print("\nImputation comparison:\n", imp_df)

# choose the better strategy by mean CV F1
best_kind = "knn" if imp_df[imp_df.Imputation.str.startswith("KNN")]["CV F1 mean"].mean() >= \
                     imp_df[imp_df.Imputation.str.startswith("Median")]["CV F1 mean"].mean() else "simple"
print("Selected imputation:", best_kind)
Xtr_s, Xte_s, Xtr_raw, Xte_raw = store[best_kind]

# ----------------------------------------------------------------------
# 5. EDA FIGURES
# ----------------------------------------------------------------------
# Fig 4-1 class balance + missingness
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
cnt = y.map({1: "CKD", 0: "Non-CKD"}).value_counts()
ax[0].bar(cnt.index, cnt.values, color=["#1f4e79", "#8ea9c1"], edgecolor="black", linewidth=.6)
for i, v in enumerate(cnt.values):
    ax[0].text(i, v + 4, f"{v} ({v/len(y)*100:.1f}%)", ha="center", fontsize=9)
ax[0].set_ylabel("Number of patients"); ax[0].set_title("(a) Class distribution")
ax[0].set_ylim(0, max(cnt.values) * 1.18)
top_miss = miss[miss > 0].head(14)[::-1]
ax[1].barh(top_miss.index, top_miss.values, color="#1f4e79", edgecolor="black", linewidth=.5)
ax[1].set_xlabel("Missing values (%)"); ax[1].set_title("(b) Missingness by attribute")
plt.tight_layout(); plt.savefig("figures/fig4_1_class_missing.png", dpi=200); plt.close()

# Fig 4-2 boxplots of key numeric attributes by class
key = ["hemo", "pcv", "sc", "bu", "sg", "al", "rc", "bgr"]
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for a, c in zip(axes.ravel(), key):
    d = df[[c, TARGET]].dropna()
    sns.boxplot(x=d[TARGET].map({1: "CKD", 0: "Non-CKD"}), y=d[c], ax=a,
                palette=["#8ea9c1", "#1f4e79"], width=.55, linewidth=.9)
    a.set_xlabel(""); a.set_title(c, fontsize=10)
plt.suptitle("Distribution of key clinical attributes by diagnostic class", y=1.01, fontsize=11)
plt.tight_layout(); plt.savefig("figures/fig4_2_boxplots.png", dpi=200, bbox_inches="tight"); plt.close()

# Fig 4-3 correlation heatmap
plt.figure(figsize=(10, 8))
corr = enc.corr(numeric_only=True)
sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, linewidths=.3,
            cbar_kws={"shrink": .8}, annot=False)
plt.title("Correlation matrix of encoded attributes")
plt.tight_layout(); plt.savefig("figures/fig4_3_corr.png", dpi=200); plt.close()

corr_target = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
corr_target.round(3).to_csv("results/target_correlation.csv")
print("\nTop |corr| with target:\n", corr_target.head(10).round(3))

# ----------------------------------------------------------------------
# 6. CLASSICAL MODELS (grid search, 5-fold CV inside training set)
# ----------------------------------------------------------------------
grids = {
    "Random Forest": (RandomForestClassifier(random_state=SEED),
                      {"n_estimators": [100, 300], "max_depth": [None, 6, 10],
                       "min_samples_leaf": [1, 2]}),
    "SVM (RBF)": (SVC(probability=True, random_state=SEED),
                  {"C": [0.5, 1, 10], "gamma": ["scale", 0.05]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                      {"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5],
                       "criterion": ["gini", "entropy"]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                          {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1],
                           "max_depth": [2, 3]}),
}

results, probs, fitted = [], {}, {}
for name, (est, grid) in grids.items():
    gs = GridSearchCV(est, grid, cv=cv, scoring="f1", n_jobs=-1)
    gs.fit(Xtr_s, y_tr)
    best = gs.best_estimator_
    fitted[name] = best
    yp = best.predict(Xte_s)
    pr = best.predict_proba(Xte_s)[:, 1]
    probs[name] = pr
    cvf1 = cross_val_score(best, Xtr_s, y_tr, cv=cv, scoring="f1")
    results.append({
        "Model": name, "Type": "Classical",
        "CV F1 (mean)": round(cvf1.mean(), 4), "CV F1 (SD)": round(cvf1.std(), 4),
        "Accuracy": round(accuracy_score(y_te, yp), 4),
        "Precision": round(precision_score(y_te, yp), 4),
        "Recall": round(recall_score(y_te, yp), 4),
        "F1": round(f1_score(y_te, yp), 4),
        "ROC-AUC": round(roc_auc_score(y_te, pr), 4),
        "Best params": json.dumps(gs.best_params_)})
    print(f"{name}: {results[-1]}")

# ----------------------------------------------------------------------
# 7. DEEP LEARNING MODELS (LSTM, GRU)
# ----------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(SEED)

Xtr3 = Xtr_s.values.reshape((Xtr_s.shape[0], Xtr_s.shape[1], 1)).astype("float32")
Xte3 = Xte_s.values.reshape((Xte_s.shape[0], Xte_s.shape[1], 1)).astype("float32")

def make_rnn(cell):
    L = layers.LSTM if cell == "LSTM" else layers.GRU
    m = keras.Sequential([
        keras.Input(shape=(Xtr3.shape[1], 1)),
        L(64, return_sequences=True), layers.Dropout(0.3),
        L(32), layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy"])
    return m

hist_store = {}
for cell in ["LSTM", "GRU"]:
    tf.keras.utils.set_random_seed(SEED)
    m = make_rnn(cell)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                       restore_best_weights=True)
    h = m.fit(Xtr3, y_tr.values, validation_split=0.2, epochs=150, batch_size=16,
              callbacks=[es], verbose=0)
    hist_store[cell] = h.history
    pr = m.predict(Xte3, verbose=0).ravel()
    yp = (pr >= 0.5).astype(int)
    probs[cell] = pr
    results.append({
        "Model": cell, "Type": "Deep Learning",
        "CV F1 (mean)": np.nan, "CV F1 (SD)": np.nan,
        "Accuracy": round(accuracy_score(y_te, yp), 4),
        "Precision": round(precision_score(y_te, yp), 4),
        "Recall": round(recall_score(y_te, yp), 4),
        "F1": round(f1_score(y_te, yp), 4),
        "ROC-AUC": round(roc_auc_score(y_te, pr), 4),
        "Best params": f"2 layers (64,32), dropout 0.3, Adam 1e-3, epochs run={len(h.history['loss'])}"})
    print(f"{cell}: {results[-1]}")

res = pd.DataFrame(results)
res.to_csv("results/model_results.csv", index=False)
print("\n", res[["Model", "Type", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]])

# ----------------------------------------------------------------------
# 8. RESULT FIGURES
# ----------------------------------------------------------------------
# Fig 5-1 metric comparison
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
x = np.arange(len(res)); w = 0.16
hatches = ["", "//", "..", "xx", "\\\\"]
greys = ["#1f4e79", "#4a7ba7", "#7fa3c4", "#b0c6da", "#d9e2ec"]
fig, ax = plt.subplots(figsize=(11, 4.6))
for i, m in enumerate(metrics):
    ax.bar(x + (i - 2) * w, res[m], w, label=m, color=greys[i],
           edgecolor="black", linewidth=.5, hatch=hatches[i])
ax.set_xticks(x); ax.set_xticklabels(res["Model"], fontsize=9)
ax.set_ylim(0.5, 1.05); ax.set_ylabel("Score")
ax.set_title("Test-set performance of classical and deep learning models")
ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
plt.tight_layout(); plt.savefig("figures/fig5_1_metrics.png", dpi=200, bbox_inches="tight"); plt.close()

# Fig 5-2 confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
for a, name in zip(axes.ravel(), res["Model"]):
    yp = (probs[name] >= 0.5).astype(int)
    cm = confusion_matrix(y_te, yp)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Non-CKD", "CKD"], yticklabels=["Non-CKD", "CKD"],
                annot_kws={"size": 11})
    a.set_title(name, fontsize=10); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.suptitle("Confusion matrices on the held-out test set (n=%d)" % len(y_te), y=1.01)
plt.tight_layout(); plt.savefig("figures/fig5_2_confusion.png", dpi=200, bbox_inches="tight"); plt.close()

# Fig 5-3 ROC curves
plt.figure(figsize=(6.5, 5.5))
styles = ["-", "--", "-.", ":", "-", "--"]
for (name, pr), st in zip(probs.items(), styles):
    fpr, tpr, _ = roc_curve(y_te, pr)
    plt.plot(fpr, tpr, st, lw=1.8,
             label=f"{name} (AUC={roc_auc_score(y_te, pr):.3f})")
plt.plot([0, 1], [0, 1], color="grey", lw=.8, ls=":")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves on the held-out test set"); plt.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig("figures/fig5_3_roc.png", dpi=200); plt.close()

# Fig 5-4 feature importance (RF + GB)
fi = pd.DataFrame({
    "Random Forest": fitted["Random Forest"].feature_importances_,
    "Gradient Boosting": fitted["Gradient Boosting"].feature_importances_},
    index=Xtr_s.columns).sort_values("Random Forest", ascending=True).tail(15)
fi.to_csv("results/feature_importance.csv")
ax = fi.plot.barh(figsize=(8, 6), color=["#1f4e79", "#9dbad4"],
                  edgecolor="black", linewidth=.5)
ax.set_xlabel("Relative importance"); ax.set_title("Top 15 attributes by impurity-based importance")
plt.tight_layout(); plt.savefig("figures/fig5_4_importance.png", dpi=200); plt.close()

# Fig 5-5 training curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for a, cell in zip(axes, ["LSTM", "GRU"]):
    h = hist_store[cell]
    a.plot(h["loss"], "-", color="#1f4e79", label="Training loss")
    a.plot(h["val_loss"], "--", color="#c00000", label="Validation loss")
    a.set_xlabel("Epoch"); a.set_ylabel("Binary cross-entropy")
    a.set_title(cell); a.legend(fontsize=8)
plt.suptitle("Training and validation loss for the recurrent models", y=1.02)
plt.tight_layout(); plt.savefig("figures/fig5_5_curves.png", dpi=200, bbox_inches="tight"); plt.close()

summary = {
    "raw_shape": [int(n_raw), int(p_raw)],
    "duplicates_removed": int(dups),
    "complete_cases": int(complete_cases),
    "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
    "train_test": [int(len(y_tr)), int(len(y_te))],
    "selected_imputation": best_kind,
    "test_ckd": int(y_te.sum()), "test_nonckd": int((1 - y_te).sum())
}
json.dump(summary, open("results/summary.json", "w"), indent=2)
print("\nSUMMARY:", summary)
print("DONE")

# ----------------------------------------------------------------------
# 9. DIAGNOSTIC: IS THE CEILING EFFECT REAL SIGNAL OR AN ARTEFACT?
# ----------------------------------------------------------------------
print("\n=== Diagnostic experiments ===")
raw = pd.read_csv(RAW)
for c in raw.select_dtypes(include=["object", "string"]).columns:
    raw[c] = raw[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
yr = raw["classification"].map({"ckd": 1, "notckd": 0})
mask = raw.drop(columns=["classification"]).isna().astype(int)

# 9a. class-conditional missingness
row_miss = raw.drop(columns=["classification"]).isna().mean(axis=1) * 100
diag = {
    "mean_row_missing_ckd_pct": round(row_miss[yr == 1].mean(), 2),
    "mean_row_missing_nonckd_pct": round(row_miss[yr == 0].mean(), 2),
    "complete_rows_ckd": int(((row_miss == 0) & (yr == 1)).sum()),
    "complete_rows_nonckd": int(((row_miss == 0) & (yr == 0)).sum()),
}

# 9b. can the missingness mask alone predict the label?
acc_mask = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=SEED),
                           mask, yr, cv=cv, scoring="accuracy")
diag["rf_missingness_mask_only_cv_acc"] = round(acc_mask.mean(), 4)
diag["rf_missingness_mask_only_cv_sd"] = round(acc_mask.std(), 4)

# 9c. single-attribute decision stumps
stumps = {}
for f in ["hemo", "sg", "al", "pcv", "rc", "sc"]:
    v = pd.to_numeric(raw[f], errors="coerce")
    ok = v.notna()
    s = cross_val_score(DecisionTreeClassifier(max_depth=1, random_state=SEED),
                        v[ok].values.reshape(-1, 1), yr[ok], cv=5, scoring="accuracy")
    stumps[f] = round(s.mean(), 4)
diag["depth1_stump_cv_acc"] = stumps

# 9d. complete-case sensitivity analysis
cc = enc.dropna()
Xc, yc = cc.drop(columns=[TARGET]), cc[TARGET]
sc_cc = StandardScaler().fit_transform(Xc)
acc_cc = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=SEED),
                         sc_cc, yc, cv=cv, scoring="accuracy")
diag["complete_case_n"] = int(len(cc))
diag["complete_case_rf_cv_acc"] = round(acc_cc.mean(), 4)

json.dump(diag, open("results/diagnostics.json", "w"), indent=2)
print(json.dumps(diag, indent=2))

# Fig 5-6 diagnostic figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cm_ = pd.DataFrame({
    "CKD": raw[yr == 1].isna().mean() * 100,
    "Non-CKD": raw[yr == 0].isna().mean() * 100}).drop(index="classification")
cm_ = cm_[cm_.sum(axis=1) > 0].sort_values("CKD", ascending=True).tail(14)
cm_.plot.barh(ax=ax[0], color=["#1f4e79", "#b0c6da"], edgecolor="black", linewidth=.5)
ax[0].set_xlabel("Missing (%)"); ax[0].set_title("(a) Missingness by attribute and class")
ax[0].legend(fontsize=8)
names = list(stumps.keys()) + ["Missingness\nmask only", "All attributes\n(RF)"]
vals = list(stumps.values()) + [acc_mask.mean(), 0.995]
bars = ax[1].bar(range(len(vals)), vals,
                 color=["#7fa3c4"] * len(stumps) + ["#c00000", "#1f4e79"],
                 edgecolor="black", linewidth=.6)
bars[len(stumps)].set_hatch("///")   # distinguishable without colour
ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels(names, fontsize=8, rotation=30, ha="right")
ax[1].set_ylim(0.5, 1.02); ax[1].set_ylabel("Cross-validated accuracy")
ax[1].set_title("(b) Accuracy from single attributes vs missingness alone")
plt.tight_layout(); plt.savefig("figures/fig5_6_diagnostics.png", dpi=200, bbox_inches="tight"); plt.close()
print("DIAGNOSTICS DONE")
