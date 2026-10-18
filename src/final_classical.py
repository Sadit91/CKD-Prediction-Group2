"""
Final-stage experiments for the classical models (Group 2, HIT401, CDU).

E1  Nested 5x5 repeated stratified CV of RF, SVM, DT, GB
    (inner 3-fold grid search on F1, so no tuning sees the outer test fold).
E2  Leakage-controlled re-evaluation with fixed interim hyper-parameters:
      a) imputed clinical values (baseline)
      b) imputed values + 0/1 missingness indicators
      c) missingness indicators only (no clinical values)
      d) complete records only (n = 158)
      e) distribution-shift tests: train on complete records -> test on
         incomplete records, and the reverse
E3  SHAP attribution (Random Forest): full data, complete-case data, and
    full data with missingness indicators
E4  SMOTE applied inside each training fold
E5  Per-attribute test of whether missingness depends on the diagnosis
    (Fisher's exact test, odds ratio, Bonferroni correction)

Outputs: results/final/*.csv, results/final/*.json
Runtime: about 10 minutes on a 2-core CPU.
"""
import os, json, warnings, time
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr, norm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
import shap
from ckd_common import (SEED, RES, CLASSICAL, STAGE1_PARAMS, GRIDS, load_clean, prep,
                        folds, metrics, make_classical)

warnings.filterwarnings("ignore")
X, y = load_clean()
FOLDS = folds(X, y)
MISS_COLS = [c for c in X.columns if X[c].isna().any()]
t_start = time.time()


def save(df, name):
    df.to_csv(os.path.join(RES, name), index=False)
    print(f"  saved {name} ({len(df)} rows)  [{time.time() - t_start:.0f}s]", flush=True)


# ---------------------------------------------------------------- E1
print("E1 nested repeated CV (classical)")
rows = []
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
for k, (tr, te) in enumerate(FOLDS):
    Xtr, Xte = prep(X.iloc[tr], X.iloc[te])
    for name in CLASSICAL:
        gs = GridSearchCV(make_classical(name), GRIDS[name], cv=inner, scoring="f1", n_jobs=1)
        gs.fit(Xtr, y.iloc[tr])
        prob = gs.best_estimator_.predict_proba(Xte)[:, 1]
        rows.append({"model": name, "fold": k, **metrics(y.iloc[te], prob),
                     "params": json.dumps(gs.best_params_)})
save(pd.DataFrame(rows), "cv_folds_classical.csv")


# ---------------------------------------------------------------- E2
def run_condition(label, Xc, yc, fl, indicators=False, only_indicators=False):
    out = []
    for k, (tr, te) in enumerate(fl):
        if only_indicators:
            Xtr = Xc.iloc[tr][MISS_COLS].isna().astype(float)
            Xte = Xc.iloc[te][MISS_COLS].isna().astype(float)
        else:
            Xtr, Xte = prep(Xc.iloc[tr], Xc.iloc[te], indicators=indicators,
                            indicator_cols=MISS_COLS)
        for name in CLASSICAL:
            m = make_classical(name, STAGE1_PARAMS[name]).fit(Xtr, yc.iloc[tr])
            out.append({"condition": label, "model": name, "fold": k,
                        **metrics(yc.iloc[te], m.predict_proba(Xte)[:, 1])})
    return out


print("E2 leakage-controlled conditions")
rows = []
rows += run_condition("Imputed clinical values", X, y, FOLDS)
rows += run_condition("Imputed values + missingness indicators", X, y, FOLDS, indicators=True)
rows += run_condition("Missingness indicators only", X, y, FOLDS, only_indicators=True)
cc = X.notna().all(axis=1)
Xcc, ycc = X[cc].reset_index(drop=True), y[cc].reset_index(drop=True)
rows += run_condition("Complete records only (n=158)", Xcc, ycc, folds(Xcc, ycc))
save(pd.DataFrame(rows), "conditions_folds.csv")

print("E2e distribution-shift tests")
rows = []
for label, trm in [("Train complete -> test incomplete", cc), ("Train incomplete -> test complete", ~cc)]:
    Xtr, Xte = prep(X[trm], X[~trm])
    for name in CLASSICAL:
        m = make_classical(name, STAGE1_PARAMS[name]).fit(Xtr, y[trm])
        rows.append({"test": label, "model": name, "n_train": int(trm.sum()),
                     "n_test": int((~trm).sum()), "ckd_in_test": int(y[~trm].sum()),
                     **metrics(y[~trm], m.predict_proba(Xte)[:, 1])})
save(pd.DataFrame(rows), "shift_tests.csv")


# ---------------------------------------------------------------- E3
print("E3 SHAP attribution")
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)


def rf_shap(Xtr, ytr, Xexplain):
    rf = make_classical("Random Forest", STAGE1_PARAMS["Random Forest"]).fit(Xtr, ytr)
    sv = shap.TreeExplainer(rf).shap_values(Xexplain)
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
    return sv


# a) full data: fitted on the interim training partition, explained on all 400 records
Xtr, Xall = prep(X.iloc[idx_tr], X)
sv_full = rf_shap(Xtr, y.iloc[idx_tr], Xall)
np.save(os.path.join(RES, "shap_values_full.npy"), sv_full)
Xall.to_csv(os.path.join(RES, "shap_features_full.csv"), index=False)
# b) complete-case data: fitted and explained on the 158 complete records
Xcc_s, _ = prep(Xcc, Xcc)
sv_cc = rf_shap(Xcc_s, ycc, Xcc_s)
# c) full data with missingness indicators
Xtr_i, Xall_i = prep(X.iloc[idx_tr], X, indicators=True, indicator_cols=MISS_COLS)
sv_ind = rf_shap(Xtr_i, y.iloc[idx_tr], Xall_i)

imp = pd.DataFrame({"feature": X.columns,
                    "mean_abs_shap_full": np.abs(sv_full).mean(0),
                    "mean_abs_shap_complete_case": np.abs(sv_cc).mean(0)})
imp["share_full"] = imp.mean_abs_shap_full / imp.mean_abs_shap_full.sum()
imp["share_complete_case"] = imp.mean_abs_shap_complete_case / imp.mean_abs_shap_complete_case.sum()
imp["rank_full"] = imp.mean_abs_shap_full.rank(ascending=False).astype(int)
imp["rank_complete_case"] = imp.mean_abs_shap_complete_case.rank(ascending=False).astype(int)
save(imp.sort_values("rank_full"), "shap_importance.csv")
ind_abs = pd.Series(np.abs(sv_ind).mean(0), index=Xall_i.columns)
is_ind = ind_abs.index.str.endswith("_missing")
rho, p_rho = spearmanr(imp.mean_abs_shap_full, imp.mean_abs_shap_complete_case)
top_ind = ind_abs[is_ind].sort_values(ascending=False)
shap_summary = {
    "spearman_rank_full_vs_complete_case": round(float(rho), 3),
    "spearman_p": float(p_rho),
    "top5_full": imp.sort_values("rank_full").feature.head(5).tolist(),
    "top5_complete_case": imp.sort_values("rank_complete_case").feature.head(5).tolist(),
    "indicator_share_of_total_attribution": round(float(ind_abs[is_ind].sum() / ind_abs.sum()), 4),
    "top_indicators": {k: round(float(v / ind_abs.sum()), 4) for k, v in top_ind.head(5).items()},
    "indicator_ranks_in_top10": [k for k in ind_abs.sort_values(ascending=False).head(10).index if k.endswith("_missing")],
}
json.dump(shap_summary, open(os.path.join(RES, "shap_summary.json"), "w"), indent=2)
print(json.dumps(shap_summary, indent=2))


# ---------------------------------------------------------------- E4
print("E4 SMOTE inside training folds")
rows = []
for k, (tr, te) in enumerate(FOLDS):
    Xtr, Xte = prep(X.iloc[tr], X.iloc[te])
    Xs, ys = SMOTE(random_state=SEED).fit_resample(Xtr, y.iloc[tr])
    for name in CLASSICAL:
        for label, (a, b) in [("No resampling", (Xtr, y.iloc[tr])), ("SMOTE in training fold", (Xs, ys))]:
            m = make_classical(name, STAGE1_PARAMS[name]).fit(a, b)
            rows.append({"resampling": label, "model": name, "fold": k,
                         **metrics(y.iloc[te], m.predict_proba(Xte)[:, 1])})
save(pd.DataFrame(rows), "smote_folds.csv")


# ---------------------------------------------------------------- E5
print("E5 missingness vs diagnosis, per attribute")
rows = []
for c in MISS_COLS:
    m = X[c].isna()
    a, b = int((m & (y == 1)).sum()), int((~m & (y == 1)).sum())   # CKD: missing, present
    cc_, d = int((m & (y == 0)).sum()), int((~m & (y == 0)).sum())  # non-CKD: missing, present
    _, p = fisher_exact([[a, b], [cc_, d]])
    # odds ratio with Haldane-Anscombe correction and Woolf 95% CI
    A, B, C, D = a + .5, b + .5, cc_ + .5, d + .5
    lor = np.log(A * D / (B * C)); se = np.sqrt(1 / A + 1 / B + 1 / C + 1 / D)
    rows.append({"attribute": c, "missing_ckd_pct": round(100 * a / (a + b), 2),
                 "missing_nonckd_pct": round(100 * cc_ / (cc_ + d), 2),
                 "odds_ratio": round(float(np.exp(lor)), 2),
                 "or_ci_low": round(float(np.exp(lor - 1.96 * se)), 2),
                 "or_ci_high": round(float(np.exp(lor + 1.96 * se)), 2),
                 "p_value": p, "p_bonferroni": min(1.0, p * len(MISS_COLS))})
mt = pd.DataFrame(rows).sort_values("odds_ratio", ascending=False)
mt["significant_bonferroni_0.05"] = mt.p_bonferroni < 0.05
save(mt, "missingness_tests.csv")
print(f"{int(mt['significant_bonferroni_0.05'].sum())} of {len(mt)} attributes significant after Bonferroni")
print("ALL DONE in %.0fs" % (time.time() - t_start))
