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

Three algorithms are trained in parallel for later comparison:

1. **HistGradientBoostingRegressor** (scikit-learn)
   - A histogram-based GBM that bins continuous features before splitting.
   - Natively handles missing values (NaN inputs do not require imputation).
   - Memory-efficient for large datasets due to histogram compression.
   - Fastest training of the three on large tabular datasets.

2. **XGBoost** (xgboost library)
   - "eXtreme Gradient Boosting" — industry standard for tabular ML.
   - Regularization terms (L1/L2) reduce overfitting on noisy financial data.
   - `n_jobs=-1` enables multi-threaded tree building.
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

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance


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
        `n_jobs=-1` tells XGBoost and LightGBM to use all available CPU cores.

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
            # - n_jobs=-1 = use all CPU cores for parallel tree building
            # - random_state=42 = deterministic results
            return XGBRegressor(
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror',
                enable_categorical=True
            )

        elif algo_name == 'lightgbm':
            # LGBMRegressor:
            # - Leaf-wise tree growth = faster convergence than level-wise
            # - verbose=-1 = suppress iteration log spam
            # - n_jobs=-1 = all CPU cores
            return LGBMRegressor(
                random_state=42,
                n_jobs=-1,
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
                X_train = X[valid_idx]
                y_train = y_target[valid_idx]

                # Instantiate a fresh, unfitted model
                model = self._get_base_estimator(algo)

                # HistGradientBoosting cannot handle very high-cardinality
                # categorical dtypes, so we feed it integer codes instead.
                # XGBoost and LightGBM keep the categorical dtype.
                X_model = X_train.copy()

                # Convert string columns to pandas category dtype so that
                # LightGBM can use native categorical splits (more efficient
                # and accurate than one-hot encoding for high-cardinality features)
                cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
                if algo == 'hist_gb':
                    for col in cat_cols:
                        X_model[col] = pd.Categorical(X_model[col].astype(str)).codes.astype('int32')
                else:
                    for col in cat_cols:
                        X_model[col] = X_model[col].astype(str).astype('category')

                # Fit: this is where actual learning happens
                model.fit(X_model, y_train)

                # Store the fitted model for later inference
                self.models[algo][h] = model

                # --- Permutation Feature Importance ---
                # n_repeats=5: each feature is shuffled 5 times, results averaged
                # n_jobs=-1: parallelized across CPU cores
                result = permutation_importance(
                    model, X_model, y_train,
                    n_repeats=5,
                    random_state=42,
                    n_jobs=-1
                )

                # Build a readable DataFrame sorted by importance descending
                imp_df = pd.DataFrame({
                    'feature': X_model.columns,
                    'importance': result.importances_mean  # average over 5 shuffles
                }).sort_values(by='importance', ascending=False)

                self.feature_importance[algo][h] = imp_df

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
