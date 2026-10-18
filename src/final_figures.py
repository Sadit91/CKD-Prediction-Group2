"""
Statistics and figures for the final report (Group 2, HIT401, CDU).

Reads the fold-level outputs of final_classical.py and final_deep.py and writes
  results/final/cv_summary.csv          mean and corrected 95% CI per model
  results/final/cv_tests.csv            corrected resampled t-tests vs Random Forest
  results/final/conditions_summary.csv  leakage-controlled conditions
  results/final/smote_summary.csv       SMOTE vs no resampling
  results/final/holdout_ci.csv          Wilson CIs and exact McNemar tests (hold-out)
  figures/fig3_1_design.png, fig5_3_cv.png, fig5_5_missing_or.png,
  figures/fig5_6_conditions.png, fig5_7_shap.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from ckd_common import RES, FIG, ROOT

NAVY, MID, LIGHT, PALE, RED, GREY = "#1f4e79", "#4a7ba7", "#9dbad4", "#d9e2ec", "#c00000", "#6b6b6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10})
ORDER = ["Random Forest", "SVM (RBF)", "Gradient Boosting", "Decision Tree", "MLP", "GRU", "LSTM"]
N_TR, N_TE = 320, 80          # 5-fold split of 400 records


def corrected_ci(x, alpha=0.05):
    """Mean and 95% CI using the Nadeau-Bengio corrected variance for k resamples."""
    x = np.asarray(x, float); k = len(x)
    var = np.var(x, ddof=1) * (1 / k + N_TE / N_TR)
    h = stats.t.ppf(1 - alpha / 2, k - 1) * np.sqrt(var)
    return x.mean(), max(0.0, x.mean() - h), min(1.0, x.mean() + h)


def corrected_ttest(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); k = len(d)
    var = np.var(d, ddof=1) * (1 / k + N_TE / N_TR)
    if var == 0:
        return np.nan, 1.0
    t = d.mean() / np.sqrt(var)
    return t, 2 * stats.t.sf(abs(t), k - 1)


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return max(0, c - h), min(1, c + h)


# ------------------------------------------------------------ repeated CV
cvc = pd.read_csv(os.path.join(RES, "cv_folds_classical.csv"))
cvd = pd.read_csv(os.path.join(RES, "cv_folds_deep.csv"))
cv = pd.concat([cvc, cvd], ignore_index=True)
metrics = ["accuracy", "precision", "recall", "specificity", "f1", "auc", "brier"]
rows = []
for m in ORDER:
    sub = cv[cv.model == m].sort_values("fold")
    r = {"model": m, "folds": len(sub)}
    for met in metrics:
        mu, lo, hi = corrected_ci(sub[met])
        r[met], r[met + "_lo"], r[met + "_hi"] = mu, lo, hi
    r["missed_ckd_total"] = int(sub.fn.sum()); r["false_alarm_total"] = int(sub.fp.sum())
    if "epochs" in sub:
        r["median_epochs"] = sub.epochs.median()
    rows.append(r)
cvs = pd.DataFrame(rows)
cvs.round(4).to_csv(os.path.join(RES, "cv_summary.csv"), index=False)

rf = cv[cv.model == "Random Forest"].sort_values("fold")
tests = []
for m in ORDER[1:]:
    sub = cv[cv.model == m].sort_values("fold")
    for met in ["accuracy", "recall", "f1"]:
        t, p = corrected_ttest(rf[met].values, sub[met].values)
        tests.append({"comparison": f"Random Forest vs {m}", "metric": met,
                      "mean_difference": rf[met].mean() - sub[met].mean(), "t": t, "p": p})
tests = pd.DataFrame(tests)
tests.round(4).to_csv(os.path.join(RES, "cv_tests.csv"), index=False)

# ------------------------------------------------------------ hold-out CIs + McNemar
ho = pd.read_csv(os.path.join(ROOT, "results", "model_results.csv"))
rows = []
for _, r in ho.iterrows():
    correct = int(round(r.Accuracy * 80)); tp = int(round(r.Recall * 50))
    lo, hi = wilson(correct, 80); rlo, rhi = wilson(tp, 50)
    errors = 80 - correct
    # Random Forest made no errors, so every discordant pair favours RF: exact two-sided p = 2 * 0.5^b
    p_mc = 1.0 if errors == 0 else min(1.0, 2 * 0.5 ** errors)
    rows.append({"model": r.Model, "correct": correct, "accuracy": r.Accuracy, "acc_ci_lo": lo,
                 "acc_ci_hi": hi, "recall": r.Recall, "rec_ci_lo": rlo, "rec_ci_hi": rhi,
                 "errors": errors, "mcnemar_p_vs_rf": p_mc})
pd.DataFrame(rows).round(4).to_csv(os.path.join(RES, "holdout_ci.csv"), index=False)

# ------------------------------------------------------------ conditions, SMOTE
cond = pd.read_csv(os.path.join(RES, "conditions_folds.csv"))
cs = cond.groupby(["condition", "model"]).agg(
    accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), recall=("recall", "mean"),
    specificity=("specificity", "mean"), f1=("f1", "mean"), auc=("auc", "mean")).reset_index()
cs.round(4).to_csv(os.path.join(RES, "conditions_summary.csv"), index=False)
sm = pd.read_csv(os.path.join(RES, "smote_folds.csv"))
sms = sm.groupby(["resampling", "model"])[["accuracy", "recall", "specificity", "f1", "auc"]].mean().reset_index()
sms.round(4).to_csv(os.path.join(RES, "smote_summary.csv"), index=False)
shift = pd.read_csv(os.path.join(RES, "shift_tests.csv"))

# ============================================================ FIGURES
# Fig 3-1 research design
fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6)
def box(x, y, w, h, title, body, fc=PALE, ec=NAVY, tc=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=8.6, weight="bold", color=tc)
    ax.text(x + w / 2, y + h - 0.5, body, ha="center", va="top", fontsize=7.4, color="#222222", linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.1))
top = [("1  Data", "UCI CKD dataset\n400 records, 24 attributes\n250 CKD / 150 non-CKD"),
       ("2  Cleaning", "strip tab/space noise\ncoerce text numerics\nencode binary attributes"),
       ("3  Partition", "stratified 80/20 hold-out\n(320 / 80) and\n5 x 5 repeated CV folds"),
       ("4  Preprocess", "median / mode imputation\nand scaling fitted on\ntraining rows only"),
       ("5  Models", "RF, SVM, DT, GB\nLSTM, GRU\nMLP (tabular DL check)")]
for i, (t, b) in enumerate(top):
    x = 0.15 + i * 2.17
    box(x, 2.25, 1.95, 1.2, t, b)
    if i < 4: arrow(x + 1.95, 2.85, x + 2.17, 2.85)
bottom = [("6  Evaluate", "hold-out once + nested\nrepeated CV; recall first;\nWilson / corrected CIs"),
          ("7  Diagnose", "single-attribute stumps\nmissingness-only model\nFisher tests per attribute"),
          ("8  Control leakage", "indicators, complete cases,\ntrain/test shift between\ncomplete and incomplete"),
          ("9  Interpret", "SHAP on full vs\ncomplete-case data;\nSMOTE sensitivity")]
for i, (t, b) in enumerate(bottom):
    x = 1.2 + i * 2.17
    box(x, 0.35, 1.95, 1.2, t, b, fc="#ffffff" if i else PALE, ec=RED if i in (1, 2) else NAVY,
        tc=RED if i in (1, 2) else NAVY)
    if i < 3: arrow(x + 1.95, 0.95, x + 2.17, 0.95)
x5 = 0.15 + 4 * 2.17 + 0.975; x6 = 1.2 + 0.975
ax.plot([x5, x5, x6], [2.25, 1.9, 1.9], color=GREY, lw=1.1)
arrow(x6, 1.9, x6, 1.55)
ax.text(10.95, 0.02, "red = final-stage experiments on the\nsource of the high scores", ha="right", va="bottom",
        fontsize=7, color=RED, style="italic")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_1_design.png"), dpi=200, bbox_inches="tight"); plt.close()

# Fig 5-3 repeated CV with corrected CIs
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for a, met, title in zip(axes, ["recall", "f1"], ["(a) Recall on the CKD class", "(b) F1 score"]):
    for i, m in enumerate(ORDER):
        r = cvs[cvs.model == m].iloc[0]
        deep = m in ("MLP", "GRU", "LSTM")
        a.errorbar(r[met], i, xerr=[[r[met] - r[met + "_lo"]], [r[met + "_hi"] - r[met]]], fmt="o" if not deep else "s",
                   color=NAVY if not deep else (RED if m != "MLP" else MID), ms=6, capsize=3, lw=1.3)
        a.text(r[met + "_lo"] - 0.004, i, f"{r[met]:.3f}", va="center", ha="right", fontsize=7.5, color="#333333")
    a.set_title(title, loc="left"); a.set_xlim(0.84, 1.006); a.grid(axis="x", color="#e6e6e6", lw=.6)
    a.set_xlabel("Mean over 25 outer folds with corrected 95% CI")
axes[0].set_yticks(range(len(ORDER))); axes[0].set_yticklabels(ORDER); axes[0].invert_yaxis()
axes[0].axhline(3.5, color="#bbbbbb", lw=.8, ls="--")
axes[1].axhline(3.5, color="#bbbbbb", lw=.8, ls="--")
axes[1].text(0.845, 3.62, "deep learning below line  (circle = classical, square = deep)", fontsize=7, color=GREY, va="top")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_3_cv.png"), dpi=200, bbox_inches="tight"); plt.close()

# Fig 5-5 missingness odds ratios
mt = pd.read_csv(os.path.join(RES, "missingness_tests.csv")).sort_values("odds_ratio")
fig, ax = plt.subplots(figsize=(7.2, 6.2))
for i, (_, r) in enumerate(mt.iterrows()):
    sig = r["significant_bonferroni_0.05"]
    ax.plot([r.or_ci_low, r.or_ci_high], [i, i], color=NAVY if sig else LIGHT, lw=1.6)
    ax.plot(r.odds_ratio, i, "o" if sig else "o", ms=6, mfc=NAVY if sig else "white", mec=NAVY)
    ax.text(ax.get_xlim()[1] if False else 400, i, f"{r.missing_ckd_pct:.0f}% vs {r.missing_nonckd_pct:.0f}%",
            va="center", fontsize=7.2, color="#333333")
ax.axvline(1, color=RED, lw=1, ls="--"); ax.set_xscale("log")
ax.set_yticks(range(len(mt))); ax.set_yticklabels(mt.attribute)
ax.set_xlim(0.02, 1500); ax.set_xlabel("Odds ratio of a value being missing in CKD vs non-CKD records (log scale, 95% CI)")
ax.text(400, len(mt) - 0.2, "missing: CKD vs non-CKD", fontsize=7.2, weight="bold", va="bottom")
ax.set_title("Missingness by attribute and diagnosis (filled = significant after Bonferroni)", loc="left", fontsize=9.5)
ax.grid(axis="x", color="#eeeeee", lw=.6)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_5_missing_or.png"), dpi=200, bbox_inches="tight"); plt.close()

# Fig 5-6 leakage-controlled conditions + shift tests
cond_order = ["Imputed clinical values", "Imputed values + missingness indicators",
              "Complete records only (n=158)", "Missingness indicators only"]
short = ["Imputed values", "Values + indicators", "Complete records\nonly (n = 158)", "Indicators only\n(no values)"]
models4 = ["Random Forest", "SVM (RBF)", "Gradient Boosting", "Decision Tree"]
marks = ["o", "s", "^", "D"]; cols = [NAVY, MID, "#2f7d5b", "#8a6d3b"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
a = axes[0]
for j, m in enumerate(models4):
    ys = [cs[(cs.condition == c) & (cs.model == m)].accuracy.iloc[0] for c in cond_order]
    a.plot(np.arange(4) + (j - 1.5) * 0.08, ys, marks[j], color=cols[j], ms=6, label=m, ls="none")
a.set_xticks(range(4)); a.set_xticklabels(short, fontsize=8); a.set_ylim(0.6, 1.02)
a.set_ylabel("Mean cross-validated accuracy"); a.grid(axis="y", color="#eeeeee", lw=.6)
a.axhline(0.625, color=GREY, ls=":", lw=1); a.text(3.45, 0.632, "majority-class rate 0.625", fontsize=7, ha="right", color=GREY)
a.set_title("(a) Same models, different information", loc="left"); a.legend(fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.12))
b = axes[1]
tests_ = ["Train complete -> test incomplete", "Train incomplete -> test complete"]
w = 0.18
for j, m in enumerate(models4):
    for t_i, tname in enumerate(tests_):
        r = shift[(shift.test == tname) & (shift.model == m)].iloc[0]
        x = t_i * 2 + (j - 1.5) * w
        b.bar(x - 0.0, r.recall, w * 0.9, color=cols[j], edgecolor="white")
        b.bar(x + 1.0 - 0.0, r.specificity, w * 0.9, color=cols[j], edgecolor="white", hatch="///", alpha=.85)
b.set_xticks([0, 1, 2, 3]); b.set_xticklabels(["recall", "specificity", "recall", "specificity"], fontsize=8)
b.text(0.5, -0.2, "trained on complete,\ntested on incomplete", transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=7.5)
b.text(2.5, -0.2, "trained on incomplete,\ntested on complete", transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=7.5)
b.set_ylim(0, 1.05); b.set_ylabel("Score"); b.grid(axis="y", color="#eeeeee", lw=.6)
b.set_title("(b) Shift between complete and incomplete records", loc="left")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_6_conditions.png"), dpi=200, bbox_inches="tight"); plt.close()

# Fig 5-7 SHAP
imp = pd.read_csv(os.path.join(RES, "shap_importance.csv")).sort_values("share_full", ascending=False).head(12)[::-1]
sv = np.load(os.path.join(RES, "shap_values_full.npy"))
Xs = pd.read_csv(os.path.join(RES, "shap_features_full.csv"))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
a = axes[0]; yy = np.arange(len(imp))
a.barh(yy + 0.2, imp.share_full, 0.38, color=NAVY, label="All 400 records (imputed)")
a.barh(yy - 0.2, imp.share_complete_case, 0.38, color=LIGHT, edgecolor=NAVY, lw=.4, hatch="///", label="158 complete records")
a.set_yticks(yy); a.set_yticklabels(imp.feature); a.set_xlabel("Share of total mean |SHAP| (Random Forest)")
a.legend(fontsize=7.5, frameon=False, loc="lower right"); a.set_title("(a) Attribution, full vs complete-case data", loc="left")
b = axes[1]
top = pd.read_csv(os.path.join(RES, "shap_importance.csv")).sort_values("share_full", ascending=False).feature.head(10).tolist()[::-1]
rng = np.random.default_rng(0)
for i, f in enumerate(top):
    j = list(Xs.columns).index(f); v = sv[:, j]; x = Xs[f].values
    q = (x - np.nanpercentile(x, 5)) / (np.nanpercentile(x, 95) - np.nanpercentile(x, 5) + 1e-9)
    b.scatter(v, i + rng.uniform(-0.28, 0.28, len(v)), c=np.clip(q, 0, 1), cmap="coolwarm", s=7, alpha=.75, lw=0)
b.axvline(0, color=GREY, lw=.8); b.set_yticks(range(len(top))); b.set_yticklabels(top)
b.set_xlabel("SHAP value (impact on predicted probability of CKD)")
b.set_title("(b) Direction of effect (blue = low value, red = high)", loc="left")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_7_shap.png"), dpi=200, bbox_inches="tight"); plt.close()

print(cvs[["model", "folds", "accuracy", "accuracy_lo", "accuracy_hi", "recall", "recall_lo", "f1", "f1_lo", "f1_hi", "auc", "missed_ckd_total", "false_alarm_total"]].round(4).to_string())
print(tests.round(4).to_string())
print(cs.round(4).to_string())
print(sms.round(4).to_string())
print(shift.round(4).to_string())
print("FIGURES DONE")
