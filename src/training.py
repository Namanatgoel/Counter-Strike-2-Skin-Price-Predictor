import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance

class MultiHorizonTrainer:
    def __init__(self, horizons: list = list(range(1, 17))):
        self.horizons = horizons
        self.models = {
            'hist_gb': {},
            'xgboost': {},
            'lightgbm': {}
        }
        self.feature_importance = {}

    def _get_base_estimator(self, algo_name):
        if algo_name == 'hist_gb':
            return HistGradientBoostingRegressor(random_state=42)
        elif algo_name == 'xgboost':
            return XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
        elif algo_name == 'lightgbm':
            return LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        raise ValueError("Unsupported algorithm")

    def train_models(self, X: pd.DataFrame, y: pd.Series):
        """Trains models directly for t+1 to t+16 horizons."""
        for algo in self.models.keys():
            self.feature_importance[algo] = {}
            for h in self.horizons:
                # Shift target for direct multi-horizon forecasting
                y_target = y.shift(-h)
                
                # Drop NaNs created by shifting
                valid_idx = y_target.notna()
                X_train, y_train = X[valid_idx], y_target[valid_idx]
                
                model = self._get_base_estimator(algo)
                model.fit(X_train, y_train)
                self.models[algo][h] = model
                
                # Calculate Permutation Importance for interpretability
                result = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=42, n_jobs=-1)
                
                imp_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': result.importances_mean
                }).sort_values(by='importance', ascending=False)
                
                self.feature_importance[algo][h] = imp_df
                
    def get_importance(self, algo_name, horizon):
        return self.feature_importance[algo_name][horizon]