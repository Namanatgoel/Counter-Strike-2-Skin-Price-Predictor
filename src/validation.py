"""
validation.py — Backtesting & Cross-Validation Module
======================================================

Purpose
-------
This module provides rigorous backtesting of the price prediction models
using **TimeSeriesSplit** cross-validation — the gold standard technique
for evaluating machine learning models on time-ordered data.

It answers the central question:
    "If I had trained this model on data available up to date D, how
     accurately would it have predicted prices between D+1 and D+H?"

Why Time-Series Cross-Validation is Critical
---------------------------------------------
Standard K-Fold cross-validation randomly shuffles data into folds.
For time-series data, this is **deeply wrong** — it creates data leakage:
the model sees tomorrow's data while "predicting" yesterday's price.

Example of leakage with standard K-Fold:
    - Fold 3 training set contains: [Jan 1, Jan 5, Jan 10, Jan 20]
    - Fold 3 test set contains: [Jan 3, Jan 7, Jan 15]
    - The model has seen Jan 5 and Jan 10 while predicting Jan 7 ← LEAK

TimeSeriesSplit uses expanding windows that strictly respect chronology:
    - Fold 1: Train[days 1-20]  → Test[days 21-24]
    - Fold 2: Train[days 1-24]  → Test[days 25-28]
    - Fold 3: Train[days 1-28]  → Test[days 29-32]
    ...

At no point does any test row appear in any training set for that fold.
This gives a realistic simulation of deploying the model in production.

Evaluation Metrics
------------------
**MAPE (Mean Absolute Percentage Error)**
    MAPE = mean(|actual - predicted| / |actual|) × 100%

    Interpretation: "On average, my predictions are off by X% from the
    true price." A MAPE of 5% means average prediction error is 5%.
    Percentage-based so it's comparable across items of different price levels.
    Weakness: undefined when actual = 0 (not an issue here since prices > 0).

**RMSE (Root Mean Squared Error)**
    RMSE = sqrt(mean((actual - predicted)²))

    Interpretation: "My predictions are typically off by $X."
    Penalizes large errors more than small ones (due to squaring).
    In the same units as the target (dollar returns), making it intuitive.
    A large RMSE relative to MAPE indicates occasional large outlier errors.

Multi-Algorithm Comparison
---------------------------
The backtest runs all three algorithms (HistGBM, XGBoost, LightGBM) on
the same data splits, enabling direct apples-to-apples comparison.
The results dict is structured as:
    benchmark[algo_name][horizon] = {'mean_mape': float, 'mean_rmse': float}

This output can be used to:
    1. Select the best-performing algorithm for production deployment.
    2. Detect if one algorithm overfits more than others on this dataset.
    3. Monitor performance across horizons (1-day vs 8-day vs 16-day).

Dependencies
------------
    numpy        — mean aggregation across folds
    pandas       — DataFrame slicing
    scikit-learn — TimeSeriesSplit, MAPE, RMSE metrics
    training     — MultiHorizonTrainer (models being evaluated)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from src.training import MultiHorizonTrainer


def backtest_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    horizons: list = list(range(1, 17)),
    n_splits: int = 5
) -> dict:
    """
    Runs a complete walk-forward backtesting evaluation across all three
    gradient boosting algorithms and all specified forecast horizons.

    This function is the primary tool for validating model quality before
    deploying predictions on real current data.

    Walk-Forward Evaluation Logic
    ------------------------------
    For each of the `n_splits` TimeSeriesSplit folds:
    1. All three algorithms are retrained on the expanding training window.
    2. For each horizon h, the model's predictions on the test window
       are compared to the actual returns shifted by h days.
    3. MAPE and RMSE are computed for each (algo, horizon, fold) combination.

    After all folds complete, MAPE and RMSE are averaged across folds to
    produce fold-aggregated benchmark metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (all items, all dates, all features).
        Must be sorted by timestamp (ascending) — TimeSeriesSplit relies
        on the implicit ordering of rows.

    y : pd.Series
        Target return Series for a SINGLE horizon (e.g., 8-day returns).
        In practice, this would be called multiple times for different
        horizons, or the y_dict from main.py could be integrated.

    horizons : list of int
        Which forecast horizons to evaluate. Default [1..16].
        Each horizon is evaluated independently.

    n_splits : int
        Number of walk-forward folds. Default 5.
        More splits = more reliable estimates but longer runtime.
        With 31 days of data and 5 splits, each fold has ~6 days.

    Returns
    -------
    dict
        Nested structure:
        {
          'lightgbm': {
            8: {'mean_mape': 0.045, 'mean_rmse': 0.003},
            ...
          },
          'xgboost': { ... },
          'hist_gb': { ... }
        }

        `mean_mape` and `mean_rmse` are the average of fold-level scores.

    Notes
    -----
    The inner `y_test_split.shift(-h)` creates the h-day-ahead target:
    if we're in the test window [day 25, day 30], shifting by -8 means
    the target is the return observed 8 days after each test day.
    Rows that fall outside the dataset (shift produces NaN) are excluded.

    Performance Warning
    -------------------
    This function is computationally intensive:
    n_splits × len(algorithms) × len(horizons) total model fits.
    For 5 splits × 3 algorithms × 16 horizons = 240 model fits.
    On the full 110MB training dataset this may take 20–60 minutes.
    For quick evaluation, use a subset of horizons or reduce n_splits.

    Example
    -------
    >>> benchmark = backtest_pipeline(X_train, y_train_dict[8], horizons=[8])
    >>> print(benchmark['lightgbm'][8])
    {'mean_mape': 0.0421, 'mean_rmse': 0.0018}
    """
    # Initialize TimeSeriesSplit: creates n_splits expanding window folds
    # where each subsequent fold's training set includes all previous folds' data
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize result storage: for each algo and horizon, collect per-fold scores
    algorithms = ['hist_gb', 'xgboost', 'lightgbm']
    results = {
        algo: {
            h: {'mape': [], 'rmse': []}
            for h in horizons
        }
        for algo in algorithms
    }

    # Create a trainer instance (will be refit for each fold)
    trainer = MultiHorizonTrainer(horizons=horizons)

    # --- Walk-Forward Evaluation Loop ---
    for fold_num, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"  Fold {fold_num + 1}/{n_splits}...")

        # Slice training and test data for this fold
        X_train_split = X.iloc[train_index]
        X_test_split  = X.iloc[test_index]
        y_train_split = y.iloc[train_index]
        y_test_split  = y.iloc[test_index]

        # Retrain all three algorithms on this fold's expanding training window.
        # We need to wrap the single-horizon y into a dict for MultiHorizonTrainer.
        y_train_dict_fold = {h: y_train_split for h in horizons}
        trainer.train_models(X_train_split, y_train_dict_fold)

        # Evaluate each algorithm and horizon on the test portion of this fold
        for algo in algorithms:
            for h in horizons:
                if h not in trainer.models.get(algo, {}):
                    continue  # model wasn't trained for this (algo, horizon)

                model = trainer.models[algo][h]

                # y_test_target: shift by -h to create the "h-days-ahead" target
                # for the test set. E.g., if test day is Jan 25 and h=8,
                # we need the return observed on Feb 2.
                y_test_target = y_test_split.shift(-h)

                # Only evaluate on rows where the future return exists
                valid_idx = y_test_target.notna()
                if valid_idx.sum() == 0:
                    continue   # no valid rows (horizon extends beyond dataset)

                X_test_eval = X_test_split[valid_idx]
                y_test_eval = y_test_target[valid_idx]

                # Generate predictions with the trained model
                preds = model.predict(X_test_eval)

                # Compute performance metrics for this fold/algo/horizon
                mape = mean_absolute_percentage_error(y_test_eval, preds)
                rmse = root_mean_squared_error(y_test_eval, preds)

                results[algo][h]['mape'].append(mape)
                results[algo][h]['rmse'].append(rmse)

    # --- Aggregate: compute mean MAPE and mean RMSE across folds ---
    benchmark = {}
    for algo in algorithms:
        benchmark[algo] = {}
        for h in horizons:
            fold_mapes = results[algo][h]['mape']
            fold_rmses = results[algo][h]['rmse']

            benchmark[algo][h] = {
                'mean_mape': np.mean(fold_mapes) if fold_mapes else float('nan'),
                'mean_rmse': np.mean(fold_rmses) if fold_rmses else float('nan')
            }

    return benchmark
