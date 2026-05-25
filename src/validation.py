import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from src.training import MultiHorizonTrainer

def backtest_pipeline(X: pd.DataFrame, y: pd.Series, horizons: list = list(range(1, 17)), n_splits: int = 5) -> dict:
    """Executes TimeSeriesSplit cross-validation and calculates MAPE and RMSE per horizon."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize metric storage
    algorithms = ['hist_gb', 'xgboost', 'lightgbm']
    results = {algo: {h: {'mape': [], 'rmse': []} for h in horizons} for algo in algorithms}
    
    trainer = MultiHorizonTrainer(horizons=horizons)
    
    for train_index, test_index in tscv.split(X):
        X_train_split, X_test_split = X.iloc[train_index], X.iloc[test_index]
        y_train_split, y_test_split = y.iloc[train_index], y.iloc[test_index]
        
        # Fit models on the expanding window
        trainer.train_models(X_train_split, y_train_split)
        
        for algo in algorithms:
            for h in horizons:
                model = trainer.models[algo][h]
                
                # Align test target for horizon
                y_test_target = y_test_split.shift(-h)
                valid_idx = y_test_target.notna()
                
                if valid_idx.sum() > 0:
                    X_test_eval = X_test_split[valid_idx]
                    y_test_eval = y_test_target[valid_idx]
                    
                    preds = model.predict(X_test_eval)
                    
                    mape = mean_absolute_percentage_error(y_test_eval, preds)
                    rmse = root_mean_squared_error(y_test_eval, preds)
                    
                    results[algo][h]['mape'].append(mape)
                    results[algo][h]['rmse'].append(rmse)
    
    # Aggregate results (Mean across folds)
    benchmark = {}
    for algo in algorithms:
        benchmark[algo] = {}
        for h in horizons:
            benchmark[algo][h] = {
                'mean_mape': np.mean(results[algo][h]['mape']),
                'mean_rmse': np.mean(results[algo][h]['rmse'])
            }
            
    return benchmark