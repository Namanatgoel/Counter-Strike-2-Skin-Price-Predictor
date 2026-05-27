"""
tuning.py — Hyperparameter Optimization Module
===============================================

Purpose
-------
This module finds the best hyperparameter configuration for LightGBM using
**Bayesian Optimization** via the Optuna library. Rather than exhaustively
testing every combination of parameters (Grid Search) or randomly sampling
combinations (Random Search), Optuna builds a probabilistic model of how
each hyperparameter setting affects model performance, and uses that model
to intelligently choose which configurations to test next.

Why Hyperparameter Tuning Matters
----------------------------------
A gradient boosting model's performance is highly sensitive to its
hyperparameters:

- A `learning_rate` too high → model overshoots the optimum, poor generalization.
- `num_leaves` too large → model memorizes training data (overfitting).
- `max_depth` too deep → same overfitting problem.
- `min_child_samples` too small → model fits noise in small data subsets.

The defaults (learning_rate=0.1, etc.) are educated guesses that often work
reasonably well but rarely optimally. Tuning finds the specific values that
minimize generalization error on this particular dataset.

Why Bayesian Optimization (Optuna's TPE Sampler)?
--------------------------------------------------
**Grid Search** tests all combinations: 10 values × 10 values × 10 values
= 1000 trials. Computationally prohibitive for continuous hyperparameters.

**Random Search** samples combinations randomly. Better than grid search
for high-dimensional spaces (most parameters don't matter much), but still
wasteful — it ignores information from past trials.

**Bayesian Optimization with TPE (Tree-structured Parzen Estimator)**:
- Models the probability distribution of good hyperparameter settings
  based on results from previous trials.
- Samples new configurations that are more likely to improve on the best
  result so far, while maintaining some exploration.
- Finds near-optimal settings in 20–50 trials vs 1000+ for grid search.

Why GroupKFold Instead of Standard K-Fold?
-------------------------------------------
Standard K-Fold randomly assigns rows to folds. In a time-series with
multiple items, this would put "AK-47 day 15" in the training set while
"AK-47 day 10" is in the test set — a form of data leakage, since the model
would effectively have seen "the future" of that item.

`GroupKFold(n_splits=5)` ensures that all observations for a given item
are either entirely in training or entirely in the test fold. This gives
a realistic estimate of how well the model generalizes to items it has
never seen during training.

Parameters Being Tuned
-----------------------
- **learning_rate** [1e-3, 0.1, log-scale]: Step size for gradient descent.
  Smaller = more stable but slower training. Searched on log scale because
  the effect of going from 0.001 to 0.01 is much larger than 0.09 to 0.1.

- **num_leaves** [20, 150]: Maximum number of leaves in a single tree.
  Controls model complexity. Too large → overfitting.

- **max_depth** [3, 15]: Maximum depth of each tree. Related to num_leaves.

- **min_child_samples** [10, 100]: Minimum number of data points required
  in a leaf node. Higher values → smoother, less overfitted model.

Dependencies
------------
    optuna       — Bayesian hyperparameter optimization framework
    pandas       — DataFrame slicing operations
    numpy        — Mean computation for RMSE aggregation
    scikit-learn — GroupKFold, root_mean_squared_error
    lightgbm     — LGBMRegressor (the model being tuned)
"""

import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from lightgbm import LGBMRegressor

# Suppress Optuna's per-trial logging — the progress bar is enough
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """
    Wraps Optuna to find the best LightGBM hyperparameters using
    Bayesian optimization with group-aware cross-validation.

    The optimization target is minimizing mean cross-validated RMSE
    of the 8-day return prediction. Lower RMSE = better generalization.

    Attributes
    ----------
    n_trials : int
        Number of hyperparameter configurations Optuna will test.
        Higher = more thorough search but longer runtime.
        20 trials is a reasonable default for quick iteration;
        100+ trials for production-grade tuning.

    horizon : int
        The forecast horizon (days ahead) for which tuning is performed.
        Tuning is done for the primary 8-day horizon; other horizons
        use the same best parameters for simplicity.

    best_params : dict or None
        Populated after `optimize()` is called. Contains the parameter
        names and values that achieved the lowest cross-validated RMSE.

    study : optuna.Study or None
        The Optuna study object. Useful for inspecting trial history,
        plotting optimization progress, or resuming a paused search.

    Example Usage
    -------------
    >>> optimizer = HyperparameterOptimizer(n_trials=20, horizon=8)
    >>> best_params = optimizer.optimize(X_features, y_8d_returns, groups=item_names)
    >>> best_model = optimizer.get_best_model(X_features, y_8d_returns)
    """

    def __init__(self, n_trials: int = 20, horizon: int = 8):
        """
        Parameters
        ----------
        n_trials : int
            How many hyperparameter configurations to test.
            Each trial trains 5 cross-validation folds.
            Total model fits = n_trials × 5.
            Default of 20 gives 100 model fits — fast but sufficient.

        horizon : int
            Forecast horizon for which these hyperparameters will be used.
            Stored for reference but does not affect the optimization logic.
        """
        self.n_trials = n_trials
        self.horizon = horizon
        self.best_params = None   # set after optimize() completes
        self.study = None         # set after optimize() completes

    def _objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series
    ) -> float:
        """
        The objective function evaluated by Optuna for each trial.

        Optuna calls this function `n_trials` times, each time passing a
        `trial` object that suggests different hyperparameter values.
        This function trains and cross-validates a LightGBM model with
        those parameters and returns the mean RMSE — Optuna then uses
        that result to inform its next suggestion.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object. `.suggest_*()` methods propose
            hyperparameter values within the specified search bounds.

        X : pd.DataFrame
            Feature matrix (same as used in training.py).

        y : pd.Series
            Target returns for the 8-day horizon.

        groups : pd.Series
            Item names (e.g., 'AK-47 | Redline') — used by GroupKFold
            to ensure all rows of the same item go to the same fold.

        Returns
        -------
        float
            Mean RMSE across 5 cross-validation folds for this trial's
            hyperparameter configuration. Optuna minimizes this value.

        Notes on `suggest_float` with `log=True`
        ------------------------------------------
        `trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)` samples
        learning_rate in log-space: values like 0.001, 0.003, 0.01, 0.03, 0.1
        are equally likely to be sampled. Without `log=True`, most samples
        would cluster near the midpoint (0.05), rarely exploring the
        potentially important low end (0.001–0.01).
        """
        # --- Define the search space for this trial ---
        params = {
            'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves':       trial.suggest_int('num_leaves', 20, 150),
            'max_depth':        trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            # Fixed params — not part of the search space
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # --- GroupKFold: 5 folds, items stay together ---
        gkf = GroupKFold(n_splits=5)
        rmse_scores = []

        # Remove rows where target is NaN (end-of-dataset future rows)
        valid_idx = y.notna()
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        groups_valid = groups[valid_idx]

        # Edge case: all targets are NaN (should not happen with proper data)
        if len(X_valid) == 0:
            return float('inf')   # return worst possible score

        # --- Run 5-fold cross-validation ---
        for train_idx, test_idx in gkf.split(X_valid, y_valid, groups=groups_valid):
            X_train = X_valid.iloc[train_idx]
            X_test  = X_valid.iloc[test_idx]
            y_train = y_valid.iloc[train_idx]
            y_test  = y_valid.iloc[test_idx]

            # Fit LightGBM with this trial's parameters
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)

            # Predict on held-out fold and measure RMSE
            preds = model.predict(X_test)
            rmse  = root_mean_squared_error(y_test, preds)
            rmse_scores.append(rmse)

        # Return mean RMSE across folds (Optuna minimizes this)
        return np.mean(rmse_scores) if rmse_scores else float('inf')

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series
    ) -> dict:
        """
        Runs the Bayesian hyperparameter search and returns the best parameters.

        Creates an Optuna Study with the TPE sampler (Tree-structured Parzen
        Estimator). The study calls `_objective` `n_trials` times, each time
        using information from previous trials to make smarter suggestions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target returns for the tuning horizon.
        groups : pd.Series
            Item name for each row — passed to GroupKFold to prevent
            items from appearing in both training and test folds.

        Returns
        -------
        dict
            Best hyperparameter configuration found.
            Example: {'learning_rate': 0.023, 'num_leaves': 87, ...}

        Notes on `seed=42` in TPESampler
        -----------------------------------
        Setting a seed makes the Bayesian search reproducible: the same
        data and seed will always explore the same sequence of parameter
        configurations. This is important for reproducibility in research
        and debugging.
        """
        # TPESampler: the default and recommended sampler for Optuna.
        # seed=42 ensures reproducible optimization runs.
        sampler = optuna.samplers.TPESampler(seed=42)

        # Create a study that minimizes the objective (lower RMSE = better)
        self.study = optuna.create_study(
            sampler=sampler,
            direction='minimize'   # we want to minimize RMSE, not maximize it
        )

        # Run the optimization — lambda captures X, y, groups for each trial
        self.study.optimize(
            lambda trial: self._objective(trial, X, y, groups),
            n_trials=self.n_trials,
            show_progress_bar=False  # set True for interactive terminal use
        )

        # Store and return the best parameters
        self.best_params = self.study.best_params
        return self.best_params

    def get_best_model(self, X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
        """
        Trains a final LightGBM model on the full dataset using the best
        hyperparameters found during optimization.

        This "final model" is different from the cross-validation models:
        those were trained on subsets of data (folds) to measure performance.
        This model uses ALL available data for training, which gives it
        the best possible predictive capability.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix (training + validation combined).
        y : pd.Series
            Full target return series.

        Returns
        -------
        LGBMRegressor
            A fitted LightGBM model using the optimal hyperparameters.
            This model is passed to `plot_shap_summary()` in visualization.py
            for feature importance analysis via SHAP values.

        Raises
        ------
        ValueError
            If called before `optimize()` — `self.best_params` would be None.
        """
        if self.best_params is None:
            raise ValueError(
                "No best parameters available. "
                "Call optimize() before get_best_model()."
            )

        # Merge best params with fixed params not part of the search space
        params = self.best_params.copy()
        params.update({
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        })

        # Remove NaN rows (same cleanup as in training.py)
        valid_idx = y.notna()
        X_train = X[valid_idx]
        y_train = y[valid_idx]

        # Train on the full dataset with optimal hyperparameters
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        return model
