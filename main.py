import pandas as pd
import numpy as np
import os
import optuna
import ujson
import warnings
from src.ingestion import load_and_flatten_json, clean_and_resample
from src.features import build_internal_features, merge_external_factors
from src.training import MultiHorizonTrainer
from src.validation import backtest_pipeline
from src.inference import QuantileInferenceEngine
from src.tuning import HyperparameterOptimizer
from src.visualization import plot_forecast_intervals, plot_shap_summary, plot_volatility_regimes

warnings.filterwarnings('ignore')

def main():
    os.makedirs('output', exist_ok=True)
    
    df = load_and_flatten_json('data/model_raw_part.json')
    df = clean_and_resample(df)
    
    # 1. Advanced Feature Engineering
    df = build_internal_features(df)
    
    timestamps = df.index.unique()
    liquidity_df = pd.DataFrame({'liquidity_vol': np.random.randint(100, 10000, len(timestamps))}, index=timestamps)
    player_count_df = pd.DataFrame({'cs2_players': np.random.randint(800000, 1500000, len(timestamps))}, index=timestamps)
    df = merge_external_factors(df, liquidity_df, player_count_df)
    
    target_col = 'spot_price'
    X = df.drop(columns=['category', 'item_name', 'marketplace_provider', target_col], errors='ignore').select_dtypes(include=[np.number])
    y = df[target_col]
    
    # 2. Hyperparameter Optimization
    optimizer = HyperparameterOptimizer(n_trials=20, horizon=8)
    best_params = optimizer.optimize(X, y)
    print("Best Hyperparameters (8d horizon):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
        
    best_model = optimizer.get_best_model(X, y)
    
    horizons = [8, 16]
    trainer = MultiHorizonTrainer(horizons=horizons)
    trainer.train_models(X, y)
    
    inference_engine = QuantileInferenceEngine(algo_name='lightgbm')
    inference_engine.fit_horizon(X, y, horizon=8)
    prediction = inference_engine.predict_with_confidence(X.iloc[[-1]], horizon=8)
    
    # 3. Visualization Suite Integration
    sample_actual = df[['spot_price']].tail(50)
    sample_pred_full = pd.DataFrame({
        'P5': [prediction['P5'].iloc[0]],
        'Prediction_P50': [prediction['Prediction_P50'].iloc[0]],
        'P95': [prediction['P95'].iloc[0]]
    }, index=[prediction.index[0]])
    plot_forecast_intervals(sample_actual, sample_pred_full, item_name='CS2_Item', output_path='output/forecast_intervals.html')
    
    y_target = y.shift(-8)
    valid_idx = y_target.notna()
    X_train_shap = X[valid_idx]
    plot_shap_summary(best_model, X_train_shap, output_path='output/shap_summary.html')
    
    sample_vol = df[['spot_price', 'volatility_14']].tail(100)
    plot_volatility_regimes(sample_vol, output_path='output/volatility_regimes.html')
    
if __name__ == "__main__":
    main()