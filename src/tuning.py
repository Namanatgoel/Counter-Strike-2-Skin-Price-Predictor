import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error
from lightgbm import LGBMRegressor

class HyperparameterOptimizer:
    def __init__(self, n_trials: int = 20, horizon: int = 8):
        self.n_trials = n_trials
        self.horizon = horizon
        self.best_params = None
        self.study = None

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        tscv = TimeSeriesSplit(n_splits=5)
        rmse_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            y_train_target = y_train.shift(-self.horizon)
            valid_train_idx = y_train_target.notna()
            X_train_clean = X_train[valid_train_idx]
            y_train_clean = y_train_target[valid_train_idx]

            if len(X_train_clean) == 0:
                continue

            model = LGBMRegressor(**params)
            model.fit(X_train_clean, y_train_clean)

            y_test_target = y_test.shift(-self.horizon)
            valid_test_idx = y_test_target.notna()
            X_test_clean = X_test[valid_test_idx]
            y_test_clean = y_test_target[valid_test_idx]

            if len(X_test_clean) > 0:
                preds = model.predict(X_test_clean)
                rmse = root_mean_squared_error(y_test_clean, preds)
                rmse_scores.append(rmse)

        return np.mean(rmse_scores) if rmse_scores else float('inf')

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> dict:
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(sampler=sampler, direction='minimize')
        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=False
        )
        self.best_params = self.study.best_params
        return self.best_params

    def get_best_model(self, X: pd.DataFrame, y: pd.Series):
        if self.best_params is None:
            raise ValueError("Run optimize() first.")
        params = self.best_params.copy()
        params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
        y_target = y.shift(-self.horizon)
        valid_idx = y_target.notna()
        X_train = X[valid_idx]
        y_train = y_target[valid_idx]
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return model
