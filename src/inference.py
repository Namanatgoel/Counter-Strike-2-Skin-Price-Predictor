import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class QuantileInferenceEngine:
    """Trains quantile models to output P5, P50 (median), and P95 for confidence scoring."""
    def __init__(self, algo_name='lightgbm'):
        self.algo_name = algo_name
        self.quantiles = [0.05, 0.50, 0.95]
        self.models = {}

    def _get_quantile_estimator(self, alpha):
        if self.algo_name == 'hist_gb':
            return HistGradientBoostingRegressor(loss='quantile', quantile=alpha, random_state=42)
        elif self.algo_name == 'xgboost':
            return XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, random_state=42, n_jobs=-1)
        elif self.algo_name == 'lightgbm':
            return LGBMRegressor(objective='quantile', alpha=alpha, random_state=42, n_jobs=-1, verbose=-1)
        raise ValueError("Unsupported algorithm for quantiles")

    def fit_horizon(self, X: pd.DataFrame, y: pd.Series, horizon: int):
        """Fits three quantile models for a specific horizon."""
        y_target = y.shift(-horizon)
        valid_idx = y_target.notna()
        X_train, y_train = X[valid_idx], y_target[valid_idx]
        
        self.models[horizon] = {}
        for q in self.quantiles:
            model = self._get_quantile_estimator(q)
            model.fit(X_train, y_train)
            self.models[horizon][q] = model

    def predict_with_confidence(self, X_infer: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Predicts intervals and calculates clamped confidence score."""
        if horizon not in self.models:
            raise ValueError(f"Model for horizon {horizon} not trained.")
            
        p5 = self.models[horizon][0.05].predict(X_infer)
        p50 = self.models[horizon][0.50].predict(X_infer)
        p95 = self.models[horizon][0.95].predict(X_infer)
        
        # Calculate Confidence: 1 - ((P95 - P5) / P50)
        # Avoid division by zero
        p50_safe = np.where(p50 == 0, 1e-6, p50)
        confidence_raw = 1 - ((p95 - p5) / p50_safe)
        
        # Clamp between 0.0 and 1.0
        confidence_clamped = np.clip(confidence_raw, 0.0, 1.0)
        
        results = pd.DataFrame({
            'P5': p5,
            'Prediction_P50': p50,
            'P95': p95,
            'Confidence_Score': confidence_clamped
        }, index=X_infer.index)
        
        return results