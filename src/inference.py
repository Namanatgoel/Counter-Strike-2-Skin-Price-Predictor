"""
inference.py — Quantile Regression Inference Engine
====================================================

Purpose
-------
This module provides **probabilistic price forecasts** rather than a single
point estimate. Instead of saying "the price in 8 days will be $12.50",
it says "the price in 8 days will most likely be $12.50, but there is a
90% chance it will fall between $10.00 and $15.50."

This is achieved through **Quantile Regression** — a technique where the
model is trained to predict a specific percentile of the target distribution
rather than the mean.

The Three Quantile Models
--------------------------
For each forecast horizon, three independent models are trained:

- **P5  (5th percentile)**  — "pessimistic" lower bound.
  95% of actual prices will land above this value.

- **P50 (50th percentile)** — the median prediction.
  This is the primary price forecast used for point-estimate reporting.
  The median is more robust to outliers than the mean.

- **P95 (95th percentile)** — "optimistic" upper bound.
  95% of actual prices will land below this value.

The gap [P5, P95] is the **90% Prediction Interval** — it captures the
model's uncertainty. A narrow interval means the model is confident; a
wide interval means the future is uncertain (high volatility, sparse data).

Confidence Score Derivation
-----------------------------
The confidence score is derived from the interval width:

    Confidence = 1 - (P95 - P5) / P50

Interpretation:
- (P95 - P5) is the absolute interval width.
- Dividing by P50 normalizes it: $2 spread on a $10 item is more significant
  than a $2 spread on a $1000 item.
- Subtracting from 1 inverts the scale: narrow interval → high confidence.
- `np.clip(..., 0.0, 1.0)` clamps to [0, 1] because the formula can
  theoretically produce values outside this range if P95 < P5 (quantile
  crossing, a known pathology of quantile regression on small datasets).

Why LightGBM for Quantile Regression?
---------------------------------------
LightGBM's `objective='quantile'` with `alpha=q` minimizes the
**Pinball Loss** (also called Quantile Loss):

    L(y, ŷ, q) = q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)

When q = 0.95, the model is penalized 19× more heavily for predicting
*below* the actual value than above it, which drives the prediction towards
the 95th percentile of the conditional distribution. This is mathematically
identical to how HistGradientBoosting implements `loss='quantile'` and how
XGBoost implements `objective='reg:quantileerror'`.

Dependencies
------------
    numpy    — Vectorized arithmetic for price reconstruction and confidence
    pandas   — DataFrame construction for prediction output
    lightgbm — Quantile regression model (primary)
    xgboost  — Alternative quantile regression model
    sklearn  — HistGradientBoostingRegressor quantile alternative
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class QuantileInferenceEngine:
    """
    Trains three quantile regression models (P5, P50, P95) for a given
    forecast horizon and provides probabilistic price predictions.

    This engine separates concerns from the MultiHorizonTrainer in training.py:
    while that trainer builds point-estimate models for performance benchmarking,
    the QuantileInferenceEngine builds uncertainty-aware models for the
    final user-facing forecast output.

    Attributes
    ----------
    algo_name : str
        The underlying gradient boosting library to use for quantile models.
        One of 'lightgbm', 'xgboost', 'hist_gb'. Default 'lightgbm'.

    quantiles : list of float
        The three quantiles to model: [0.05, 0.50, 0.95].
        These represent P5, P50 (median), and P95 respectively.

    models : dict
        Nested dict: models[horizon][quantile] = fitted estimator.
        Populated by `fit_horizon()`.

    Example Usage
    -------------
    >>> engine = QuantileInferenceEngine(algo_name='lightgbm')
    >>> engine.fit_horizon(X_train, y_train_8d, horizon=8)
    >>> predictions = engine.predict_with_confidence(X_test, current_prices, horizon=8)
    >>> print(predictions.head())
          P5  Prediction_P50   P95  Confidence_Score
    ...
    """

    def __init__(self, algo_name: str = 'lightgbm'):
        """
        Parameters
        ----------
        algo_name : str
            Gradient boosting library to use. 'lightgbm' recommended for
            its speed and quantile accuracy. 'xgboost' and 'hist_gb' are
            alternatives for comparison or validation.
        """
        self.algo_name = algo_name
        self.quantiles = [0.05, 0.50, 0.95]  # P5, median, P95
        self.models = {}                       # filled by fit_horizon()

    def _get_quantile_estimator(self, alpha: float):
        """
        Returns an unfitted quantile regression estimator for the given quantile.

        Each algorithm implements quantile regression differently under the hood,
        but all expose the same sklearn-compatible `fit(X, y)` / `predict(X)` API.

        Parameters
        ----------
        alpha : float
            Target quantile in [0, 1]. E.g., 0.05 for P5, 0.95 for P95.

        Returns
        -------
        An unfitted sklearn-compatible quantile regressor.

        Implementation Notes
        --------------------
        - HistGradientBoosting: `loss='quantile'` with `quantile=alpha`
          minimizes the pinball loss function.
        - XGBoost: `objective='reg:quantileerror'` with `quantile_alpha=alpha`
          is XGBoost's native quantile regression mode (added in v1.7).
        - LightGBM: `objective='quantile'` with `alpha=alpha` is the most
          battle-tested option and typically the most accurate for this use case.
        """
        if self.algo_name == 'hist_gb':
            return HistGradientBoostingRegressor(
                loss='quantile',
                quantile=alpha,
                random_state=42
            )
        elif self.algo_name == 'xgboost':
            return XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=alpha,
                random_state=42,
                n_jobs=-1
            )
        elif self.algo_name == 'lightgbm':
            return LGBMRegressor(
                objective='quantile',
                alpha=alpha,       # LightGBM's parameter name for the target quantile
                random_state=42,
                n_jobs=-1,
                verbose=-1         # suppress training log output
            )

        raise ValueError(
            f"Unsupported algorithm '{self.algo_name}' for quantile regression."
        )

    def fit_horizon(self, X: pd.DataFrame, y_target: pd.Series, horizon: int):
        """
        Trains three quantile models (P5, P50, P95) for one forecast horizon.

        Each of the three models is trained independently on the same dataset.
        They share the same features but optimize different quantile loss
        functions, so their predictions represent different percentiles of
        the conditional price distribution.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Same format as used in training.py.
            Rows with NaN targets are automatically excluded.

        y_target : pd.Series
            Target percentage returns for the given horizon.
            y[t] = price(t + horizon) / price(t) - 1

        horizon : int
            The forecast horizon in days (e.g., 8 for 8-day-ahead prediction).
            Used as the dictionary key in `self.models`.

        Side Effects
        ------------
        Populates `self.models[horizon]` with a dict mapping each quantile
        value to its fitted model:
            {0.05: fitted_P5_model, 0.50: fitted_P50_model, 0.95: fitted_P95_model}
        """
        # Remove rows where the future return is unknown (end-of-dataset NaN)
        valid_idx = y_target.notna()
        X_train = X[valid_idx]
        y_train = y_target[valid_idx]

        self.models[horizon] = {}  # initialize storage for this horizon

        # Train one model per quantile
        for q in self.quantiles:
            model = self._get_quantile_estimator(q)
            model.fit(X_train, y_train)
            self.models[horizon][q] = model   # store fitted model at key q

    def predict_with_confidence(
        self,
        X_infer: pd.DataFrame,
        current_prices: pd.Series,
        horizon: int
    ) -> pd.DataFrame:
        """
        Generates probabilistic price predictions for the given feature matrix.

        Process
        -------
        1. Each of the three quantile models predicts a *return* (percentage
           change) for each row in X_infer.
        2. The returns are converted to absolute prices by multiplying against
           `current_prices`: predicted_price = current_price * (1 + return).
        3. A confidence score is computed from the interval width.

        Parameters
        ----------
        X_infer : pd.DataFrame
            Feature matrix for the observations to predict on.
            Must have the same columns as the training data.

        current_prices : pd.Series
            The current (most recent) spot price for each row in X_infer.
            Same index as X_infer. Used to convert return predictions to
            absolute price predictions.

        horizon : int
            Must match a horizon previously passed to `fit_horizon()`.

        Returns
        -------
        pd.DataFrame
            Same index as X_infer. Columns:
            - 'P5'              : Lower bound price (5th percentile)
            - 'Prediction_P50'  : Median price prediction (primary output)
            - 'P95'             : Upper bound price (95th percentile)
            - 'Confidence_Score': Model confidence in [0.0, 1.0]
                                  Higher = narrower interval = more confident.

        Raises
        ------
        ValueError
            If `horizon` was not fitted via `fit_horizon()`.

        Confidence Score Formula
        ------------------------
        raw_confidence = 1 - ((P95_price - P5_price) / P50_price)
        clamped        = clip(raw_confidence, 0.0, 1.0)

        A raw confidence of 1.0 means P5 == P95 (perfect certainty, impossible in practice).
        A raw confidence near 0 means the interval width equals the predicted price.
        Values < 0 (interval wider than price) are clamped to 0.
        """
        if horizon not in self.models:
            raise ValueError(
                f"No model trained for horizon={horizon}. "
                f"Call fit_horizon() first."
            )

        # --- Step 1: Predict percentage returns for each quantile ---
        # Each model's predict() returns a numpy array of return values
        p5_ret  = self.models[horizon][0.05].predict(X_infer)  # pessimistic return
        p50_ret = self.models[horizon][0.50].predict(X_infer)  # median return
        p95_ret = self.models[horizon][0.95].predict(X_infer)  # optimistic return

        # --- Step 2: Convert returns to absolute prices ---
        # If current price is $10 and predicted return is +0.05, predicted price = $10.50
        p5  = current_prices.values * (1 + p5_ret)
        p50 = current_prices.values * (1 + p50_ret)
        p95 = current_prices.values * (1 + p95_ret)

        # --- Step 3: Compute Confidence Score ---
        # Guard against division by zero if P50 is exactly 0 (pathological edge case)
        p50_safe = np.where(p50 == 0, 1e-6, p50)

        # Interval width relative to the predicted price — smaller = more confident
        confidence_raw = 1 - ((p95 - p5) / p50_safe)

        # Clamp to [0, 1]: quantile crossing (P95 < P5) would give values > 1;
        # extremely wide intervals give negative values. Both are clamped.
        confidence_clamped = np.clip(confidence_raw, 0.0, 1.0)

        # --- Step 4: Package results into a DataFrame ---
        results = pd.DataFrame({
            'P5':               p5,
            'Prediction_P50':   p50,
            'P95':              p95,
            'Confidence_Score': confidence_clamped
        }, index=X_infer.index)  # preserve original index (timestamps)

        return results
