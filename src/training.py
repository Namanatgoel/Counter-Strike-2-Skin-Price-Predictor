"""
training.py — Multi-Horizon Model Training Module
=================================================

Purpose
-------
This module trains one gradient boosting regression model per (algorithm,
horizon) combination. A "horizon" of 8 means "predict the price 8 days
from now". By training separate models for each horizon rather than a
single model that predicts all horizons at once, we use a strategy called
**Direct Multi-Step Forecasting**, which avoids error accumulation that
plagues recursive (iterated one-step) approaches.

Three algorithms are trained sequentially for later comparison:

1. **HistGradientBoostingRegressor** (scikit-learn)
   - A histogram-based GBM that bins continuous features before splitting.
   - Natively handles missing values (NaN inputs do not require imputation).
   - Memory-efficient for large datasets due to histogram compression.
   - Fastest training of the three on large tabular datasets.

2. **XGBoost** (xgboost library)
   - "eXtreme Gradient Boosting" — industry standard for tabular ML.
   - Regularization terms (L1/L2) reduce overfitting on noisy financial data.
    - `n_jobs=2` keeps thread-pool memory use bounded.
   - `objective='reg:squarederror'` optimizes the squared-error loss
     function, appropriate for continuous price return prediction.

3. **LightGBM** (lightgbm library)
   - Leaf-wise tree growth (instead of level-wise like XGBoost), which
     finds deeper patterns with fewer trees.
   - Typically trains 3–10× faster than XGBoost on the same dataset.
   - `verbose=-1` silences the per-iteration logging to keep output clean.

Why Train All Three?
--------------------
Each algorithm makes different inductive biases — assumptions about what
kinds of patterns exist in the data. Financial time-series are notoriously
non-stationary: what works in one market regime may fail in another. By
training all three and comparing their cross-validated performance
(see validation.py), the practitioner can select the best model for the
current market condition or ensemble them.

Feature Importance via Permutation Importance
----------------------------------------------
After each model is trained, we calculate Permutation Importance:
- Take the trained model's predictions on the training set.
- Randomly shuffle one feature column at a time, breaking its correlation
  with the target.
- Re-score the model; the drop in performance = how important that feature is.

This approach is model-agnostic (works for any model) and measures
*actual predictive contribution* rather than the internal split statistics
reported by tree models (which are biased towards high-cardinality features).

Direct Multi-Step Forecasting Explained
-----------------------------------------
Instead of predicting tomorrow's price and using that to predict the day
after (recursive/iterated approach), Direct Forecasting trains a completely
independent model for each target horizon:

    Model_h8: features(today) → return_in_8_days
    Model_h1: features(today) → return_in_1_day
    (etc. for each horizon in self.horizons)

The target for horizon h is:
    y = price(t+h) / price(t) - 1   (percentage return, computed in main.py)

This avoids the problem of error accumulation: in recursive forecasting,
each step's error is fed into the next prediction, compounding rapidly.
Direct forecasting makes independent predictions for each horizon.

Dependencies
------------
    pandas       — DataFrame operations
    numpy        — Array operations
    scikit-learn — HistGradientBoostingRegressor, permutation_importance
    xgboost      — XGBRegressor
    lightgbm     — LGBMRegressor
"""

import gc

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance


MAX_WORKERS = 2
MAX_HISTGB_CATEGORIES = 255


def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert object columns to category."""
    float_cols = df.select_dtypes(include=['float64']).columns
    for column in float_cols:
        df[column] = df[column].astype('float32')

    int_cols = df.select_dtypes(include=['int64']).columns
    for column in int_cols:
        df[column] = df[column].astype('int32')

    object_cols = df.select_dtypes(include=['object']).columns
    for column in object_cols:
        df[column] = df[column].astype('category')

    return df


def _normalize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force categorical inputs to use a uniform string-backed category dtype."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for column in categorical_cols:
        df[column] = df[column].astype(str).astype('category')

    return df


def _drop_high_cardinality_categoricals(df: pd.DataFrame, max_categories: int = MAX_HISTGB_CATEGORIES) -> pd.DataFrame:
    """Drop categorical columns that exceed HistGradientBoosting's category limit."""
    categorical_cols = df.select_dtypes(include=['category']).columns
    columns_to_drop = [
        column for column in categorical_cols
        if df[column].nunique(dropna=True) > max_categories
    ]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df


class MultiHorizonTrainer:
    """
    Trains and stores gradient boosting models for multiple prediction horizons
    across three different algorithm families.

    Each trained model maps today's feature vector (technical indicators,
    external factors) to the expected percentage return N days from today,
    where N ∈ self.horizons.

    Attributes
    ----------
    horizons : list of int
        Forecast horizons in days. Default [8] in the main pipeline.
        Can be extended to [1, 2, 4, 8, 16] for multi-horizon analysis.

    models : dict
        Nested dict: models[algo_name][horizon] = fitted sklearn-compatible model.
        Example: models['lightgbm'][8] = a trained LGBMRegressor instance.

    feature_importance : dict
        Nested dict: feature_importance[algo_name][horizon] = pd.DataFrame
        with columns ['feature', 'importance'], sorted descending.

    Example Usage
    -------------
    >>> trainer = MultiHorizonTrainer(horizons=[8])
    >>> trainer.train_models(X_features, y_dict)
    >>> importance_df = trainer.get_importance('lightgbm', 8)
    >>> print(importance_df.head(5))
    """

    def __init__(self, horizons: list = list(range(1, 17))):
        """
        Initializes the trainer with the desired forecast horizons.

        Parameters
        ----------
        horizons : list of int
            Each integer N means "train a model to predict N days ahead".
            Default is [1..16] but the main pipeline uses [8] due to
            data length constraints (see main.py comments).
        """
        self.horizons = horizons

        # Pre-allocate nested dicts to store one model per (algo, horizon) pair
        self.models = {
            'hist_gb':   {},   # sklearn HistGradientBoostingRegressor
            'xgboost':   {},   # XGBoost XGBRegressor
            'lightgbm':  {}    # LightGBM LGBMRegressor
        }

        # Will store permutation importance DataFrames after training
        self.feature_importance = {}

    def _get_base_estimator(self, algo_name: str):
        """
        Instantiates a fresh (unfitted) estimator for the given algorithm.

        All models use `random_state=42` for reproducibility — given the
        same data and this seed, every run produces identical results.
        `n_jobs=2` keeps the training process from over-subscribing RAM.

        Parameters
        ----------
        algo_name : str
            One of 'hist_gb', 'xgboost', 'lightgbm'.

        Returns
        -------
        An unfitted sklearn-compatible estimator.

        Raises
        ------
        ValueError
            If `algo_name` is not one of the supported values.
        """
        if algo_name == 'hist_gb':
            # HistGradientBoostingRegressor:
            # - Uses histogram-based splits (fast, memory-efficient)
            # - Natively handles NaN without preprocessing
            # - Default loss is squared error, appropriate for regression
            return HistGradientBoostingRegressor(
                random_state=42,
                categorical_features="from_dtype"
            )

        elif algo_name == 'xgboost':
            # XGBRegressor:
            # - 'reg:squarederror' = minimize (prediction - actual)²
            # - tree_method='hist' = lower-memory histogram builder
            # - n_jobs=2 = keep worker count small
            # - random_state=42 = deterministic results
            return XGBRegressor(
                random_state=42,
                n_jobs=MAX_WORKERS,
                objective='reg:squarederror',
                tree_method='hist',
                enable_categorical=True,
                max_bin=256
            )

        elif algo_name == 'lightgbm':
            # LGBMRegressor:
            # - Leaf-wise tree growth = faster convergence than level-wise
            # - verbose=-1 = suppress iteration log spam
            # - n_jobs=2 = all CPU cores is unnecessary for this workload
            return LGBMRegressor(
                random_state=42,
                n_jobs=MAX_WORKERS,
                verbose=-1
            )

        raise ValueError(
            f"Unsupported algorithm '{algo_name}'. "
            f"Choose from: 'hist_gb', 'xgboost', 'lightgbm'."
        )

    def train_models(self, X: pd.DataFrame, y_dict: dict):
        """
        Trains one model per (algorithm, horizon) combination.

        For each horizon h, the target variable y is the percentage return
        over the next h days:
            y[h] = price(t+h) / price(t) - 1

        Rows where `y[h]` is NaN (because the future price does not yet
        exist in the dataset) are automatically excluded from training.

        Also computes Permutation Importance for each fitted model to
        quantify how much each feature contributes to predictive accuracy.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Each row is one observation (one item, one day).
            Columns are the technical indicator features from features.py.
            Must NOT contain 'item_name' or 'spot_price' columns — these
            should be dropped before calling this function (done in main.py).

        y_dict : dict
            Maps horizon (int) → pd.Series of target returns.
            Example: {8: pd.Series([0.02, -0.01, ...])}
            Computed by `compute_targets()` in main.py.

        Side Effects
        -----------
        Populates `self.models[algo][horizon]` and
        `self.feature_importance[algo][horizon]` for every (algo, horizon)
        pair where the horizon is present in `y_dict`.

        Notes on Permutation Importance
        --------------------------------
        `permutation_importance` shuffles each feature column independently
        and measures how much the model's score (R²) drops. A large drop
        means the model depends heavily on that feature; a small drop means
        the feature is redundant. `n_repeats=5` shuffles each feature 5 times
        and averages the results to reduce randomness in the estimate.
        """
        for algo in self.models.keys():
            self.feature_importance[algo] = {}   # initialize dict for this algorithm

            for h in self.horizons:
                # Skip if no target was provided for this horizon
                if h not in y_dict:
                    continue

                y_target = y_dict[h]

                # Remove rows where the future price is unknown (end of dataset)
                valid_idx = y_target.notna()
                X_train = _normalize_categorical_columns(X.loc[valid_idx].copy())
                if algo == 'hist_gb':
                    X_train = _drop_high_cardinality_categoricals(X_train)
                y_train = y_target.loc[valid_idx].astype('float32', copy=False)

                model = None
                result = None
                imp_df = None

                try:
                    # Fit: this is where actual learning happens.
                    # The input frame is already optimized and categorical
                    # columns are preserved as pandas category dtype.
                    model = self._get_base_estimator(algo)
                    model.fit(X_train, y_train)

                    # Store the fitted model for later inference.
                    self.models[algo][h] = model

                    # Keep permutation importance bounded on very large frames.
                    importance_max_samples = 100_000 if len(X_train) > 100_000 else 1.0
                    result = permutation_importance(
                        model,
                        X_train,
                        y_train,
                        n_repeats=5,
                        random_state=42,
                        n_jobs=1,
                        max_samples=importance_max_samples
                    )

                    # Build a readable DataFrame sorted by importance descending.
                    imp_df = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': result.importances_mean
                    }).sort_values(by='importance', ascending=False)

                    self.feature_importance[algo][h] = imp_df
                finally:
                    del result
                    del imp_df
                    del model
                    del X_train
                    del y_train
                    gc.collect()

    def get_importance(self, algo_name: str, horizon: int) -> pd.DataFrame:
        """
        Returns the feature importance DataFrame for a trained (algo, horizon) model.

        Parameters
        ----------
        algo_name : str
            One of 'hist_gb', 'xgboost', 'lightgbm'.
        horizon : int
            The forecast horizon (must have been included in `self.horizons`
            and present in `y_dict` during training).

        Returns
        -------
        pd.DataFrame
            Columns: ['feature', 'importance']
            Sorted descending by importance. Importance values represent
            the mean drop in R² score when that feature is shuffled.

        Example
        -------
        >>> trainer.get_importance('lightgbm', 8)
            feature     importance
        0   rsi_14         0.0342
        1   volatility_14  0.0298
        2   mom_7          0.0215
        ...
        """
        return self.feature_importance[algo_name][horizon]
