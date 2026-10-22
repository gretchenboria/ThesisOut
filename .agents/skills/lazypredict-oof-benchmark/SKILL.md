---
name: lazypredict-oof-benchmark
description: Use this skill when running ~30 supervised machine learning models (LazyClassifier / LazyRegressor) with strict inside-fold preprocessing, leakage proxy pruning, and out-of-fold 95% confidence intervals.
---

# LazyPredict Out-of-Fold 30-Model Benchmark Skill

## Overview
Evaluates ~30 machine learning architectures on a real tabular dataset (`DatasetArtifact`) while preventing preprocessing leakage and target proxy inflation.

## Execution Steps
1. Strip identifier columns (`Unnamed: 0`, monotonic row counters, UUIDs, sample IDs, free-text strings).
2. Audit training split (`X_tr_audit`) for single-feature target proxies (`|r| >= 0.88`, Spearman `|rho| >= 0.90`, depth-2 decision stump CV `>= 0.92`) and multi-feature post-hoc composites, pruning any leaking columns before evaluation.
3. Fit `SimpleImputer`, `OrdinalEncoder`, and `StandardScaler` strictly inside each cross-validation fold (`StratifiedKFold`, `RepeatedStratifiedKFold`, `KFold`, `RepeatedKFold`).
4. Evaluate all ~30 models out-of-fold to compute ROC-AUC, Balanced Accuracy, Precision, Recall, F1 (or R-Squared, Adjusted R-Squared, RMSE, MAE) and empirical 95% confidence intervals.
5. Extract permutation/ensemble feature importances and compare against linear baseline (`LogisticRegression` / `Ridge`) to compute `signal_gain_delta`.
