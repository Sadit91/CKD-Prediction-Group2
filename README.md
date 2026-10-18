# Kidney Failure Prediction Using ML and DL — Group 2

HIT401 Capstone Project, Charles Darwin University (Faculty of Science and Technology)
Supervisor: Dr Asif Karim
Team: Abu Lise, Raianul Islam, Sadit Khan, Tonmoy Acharjee

This repository reproduces every table and figure in the Group 2 final report:
four classical classifiers (Random Forest, SVM, Decision Tree, Gradient Boosting)
and three neural networks (LSTM, GRU and a feed-forward MLP control) for binary
chronic kidney disease (CKD) prediction, followed by diagnostic and
leakage-controlled experiments that explain why accuracies on this benchmark
approach 100%.

## Repository structure

| Path | Contents |
|---|---|
| `data/kidney_disease.csv` | Source dataset: 400 records, 24 attributes + label (UCI Chronic Kidney Disease dataset) |
| `src/ckd_pipeline.py` | Stage 1 (interim report): cleaning, imputation comparison, EDA, hold-out training and evaluation of RF, SVM, DT, GB, LSTM, GRU, first diagnostics |
| `src/ckd_common.py` | Shared loading, cleaning, train-only preprocessing, fold generation and metrics for stage 2 |
| `src/final_classical.py` | Stage 2 experiments E1–E5 for the classical models (nested repeated CV, leakage-controlled conditions, shift tests, SHAP, SMOTE, missingness tests) |
| `src/final_deep.py` | Stage 2 repeated CV for LSTM, GRU and MLP on the same folds |
| `src/final_figures.py` | Corrected confidence intervals, significance tests and all stage 2 figures |
| `notebooks/CKD_Analysis_Group2.ipynb` | Executed notebook version of stage 1 (outputs saved) |
| `figures/` | All figures used in the report (PNG, 200 dpi) |
| `results/` | Stage 1 outputs (CSV/JSON) |
| `results/final/` | Stage 2 outputs, including fold-level results for every model |

## Environment configuration

Tested with Python 3.11 on Linux (2-core CPU, no GPU required).

```
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the report:
pandas 3.0, numpy 2.4, scikit-learn 1.8, tensorflow-cpu 2.21, shap 0.51,
imbalanced-learn 0.14, scipy 1.17, matplotlib 3.10, seaborn 0.13.

A fixed seed of 42 is set for Python, NumPy, scikit-learn and TensorFlow.
TensorFlow may print oneDNN notices; set `TF_ENABLE_ONEDNN_OPTS=0` for bit-exact
CPU results across machines.

## Execution steps

Run from the repository root, in this order:

```
python src/ckd_pipeline.py          # stage 1: ~1.5 min  -> figures/fig4_*, fig5_*, results/*.csv
cd src
python final_classical.py           # stage 2 classical: ~6 min -> results/final/*.csv
python final_deep.py                # stage 2 deep CV: ~45 min on 2 cores -> results/final/cv_folds_deep.csv
python final_figures.py             # stats + figures: <1 min -> results/final/*_summary.csv, figures/fig3_1, fig5_3_cv, fig5_5..5_7
```

`final_deep.py` appends one row per model per fold, so an interrupted run can be
restarted and will continue from where it stopped. To run one model only:
`python final_deep.py MLP`.

The dataset path defaults to `data/kidney_disease.csv`; override it with the
environment variable `CKD_DATA` if needed.

## Experimental setup and parameter settings

**Data and cleaning.** Tab and space characters are stripped from every text
value (they corrupted several categories and the label), `pcv`, `wc` and `rc` are
coerced from text to numbers, and the ten binary attributes are encoded 0/1.
No duplicates were found. 158 of 400 records are complete.

**Leakage control.** Every fitted transformation (median imputation for numeric
attributes, mode imputation for binary attributes, standard scaling) is fitted on
the training rows of each split only and then applied to the test rows.

**Partitions.**
- Hold-out: stratified 80/20 split (320 training, 80 test; 50 CKD, 30 non-CKD), `random_state=42`. The test set is used once.
- Repeated CV: 5-fold stratified CV repeated 5 times (25 outer folds), `random_state=42`, identical for every model and experiment.

**Hyper-parameters.**

| Model | Search space (5-fold grid search on F1 in stage 1; 3-fold inner search in nested CV) | Selected on hold-out training partition |
|---|---|---|
| Random Forest | n_estimators {100, 300}; max_depth {None, 6, 10}; min_samples_leaf {1, 2} | 100 trees, unlimited depth, leaf 1 |
| SVM (RBF) | C {0.5, 1, 10}; gamma {scale, 0.05} | C = 0.5, gamma = 0.05 |
| Decision Tree | max_depth {3, 5, None}; min_samples_leaf {1, 3, 5}; criterion {gini, entropy} | depth 3, leaf 1, gini |
| Gradient Boosting | n_estimators {100, 300}; learning_rate {0.05, 0.1}; max_depth {2, 3} | 300 trees, rate 0.1, depth 2 |
| LSTM / GRU | fixed: recurrent layers 64 and 32 units, dropout 0.3, Dense 16 (ReLU), sigmoid output; Adam 1e-3; batch 16; up to 150 epochs; early stopping on validation loss (patience 15, 20% validation split, best weights restored); the 24 scaled attributes are presented as a sequence of 24 steps | — |
| MLP (control) | as above with Dense 64 and 32 layers instead of recurrent layers; attributes presented as a vector | — |

The leakage-controlled conditions, shift tests, SHAP and SMOTE experiments use the
stage 1 selected hyper-parameters so that only the data condition changes.

**Statistics.** Hold-out accuracy and recall carry Wilson 95% intervals, and
models are compared with Random Forest by an exact McNemar test. Repeated-CV
means carry 95% intervals with the Nadeau–Bengio variance correction for
overlapping training sets, and pairwise comparisons use the corrected resampled
t-test. Missingness is tested per attribute with Fisher's exact test
(Bonferroni-corrected) and summarised by odds ratios with Woolf intervals.

## Data source

Rubini, L., Soundarapandian, P. and Eswaran, P. (2015) *Chronic Kidney Disease*
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020
(supplied to the group by the supervisor). The data are de-identified.

## Figures in the final report

| Report figure | File | Produced by |
|---|---|---|
| Fig. 3-1 Research design | `figures/fig3_1_design.png` | `final_figures.py` |
| Fig. 4-1 Class distribution and missingness | `figures/fig4_1_class_missing.png` | `ckd_pipeline.py` |
| Fig. 4-2 Key attributes by class | `figures/fig4_2_boxplots.png` | `ckd_pipeline.py` |
| Fig. 4-3 Correlation matrix | `figures/fig4_3_corr.png` | `ckd_pipeline.py` |
| Fig. 4-4 LSTM/GRU training curves | `figures/fig5_5_curves.png` | `ckd_pipeline.py` |
| Fig. 5-1 Confusion matrices | `figures/fig5_2_confusion.png` | `ckd_pipeline.py` |
| Fig. 5-2 ROC curves | `figures/fig5_3_roc.png` | `ckd_pipeline.py` |
| Fig. 5-3 Repeated-CV recall and F1 with CIs | `figures/fig5_3_cv.png` | `final_figures.py` |
| Fig. 5-4 Diagnostics (stumps, missingness-only) | `figures/fig5_6_diagnostics.png` | `ckd_pipeline.py` |
| Fig. 5-5 Missingness odds ratios | `figures/fig5_5_missing_or.png` | `final_figures.py` |
| Fig. 5-6 Information conditions and shift tests | `figures/fig5_6_conditions.png` | `final_figures.py` |
| Fig. 5-7 SHAP attribution | `figures/fig5_7_shap.png` | `final_figures.py` |

(File names from the interim pipeline keep their original numbering.)

## Publishing this repository

```
git init && git add . && git commit -m "Group 2 CKD prediction: code, data and results"
git branch -M main
git remote add origin https://github.com/<your-account>/CKD-Prediction-Group2.git
git push -u origin main
```
Make the repository **public** and paste its link into Appendix C of the report.
