---
name: deep-conformal-escalation
description: Use this skill when escalating a high-gain LazyPredict run to 5-stage Deep Inference verification (5-fold CV, permutation null test, top-feature ablation, Super-Learner stacking, surrogate decision thresholds, and 90% split-conformal calibration).
---

# Deep Conformal Escalation & Super-Learner Verification Skill

## Overview
Executes the 5-stage statistical verification pipeline (`deep_inference_agent.py`) on a discovered target signal.

## Execution Steps
1. **Stage 1 (Leakage & Permutation Null Audit)**: Run 5-fold leak-free CV, compare against 5 scrambled target permutations (`y_perm`), and run top-feature ablation (`X_without_top`).
2. **Stage 2 (Super-Learner Stacking)**: Train a Level-2 `StackingClassifier` or `StackingRegressor` combining `LGBM`, `XGBoost`, `ExtraTrees`, and `RandomForest`.
3. **Stage 3 (Surrogate Thresholds & Quadrant Surface)**: Fit a depth-3 decision tree on training split predictions and compute the 2x2 interaction response surface across training-split medians.
4. **Stage 4 (Split-Conformal Calibration)**: Compute 90% conformal prediction intervals (regression) or high-certainty regime accuracy (classification) on holdout test samples.
5. **Stage 5 (Counterfactual Prescription & Python Export)**: Output parameter shift prescriptions and a standalone leak-free `.py` verification script.
