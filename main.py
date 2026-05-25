import pandas as pd
import numpy as np
from src.ingestion import load_and_flatten_json, clean_and_resample
from src.features import build_internal_features, merge_external_factors
from src.training import MultiHorizonTrainer
from src.validation import backtest_pipeline
from src.inference import QuantileInferenceEngine
import warnings

warnings.filterwarnings('ignore')

def main():
    print("1. Data Ingestion...")
    # Using the provided sample file name
    df = load_and_flatten_json('data/model_raw_part.json')
    df = clean_and_resample(df)
    
    print("2. Feature Engineering...")
    df = build_internal_features(df)
    
    # Mock Exogenous Data Generation for execution demonstration
    timestamps = df.index.unique()
    liquidity_df = pd.DataFrame({'liquidity_vol': np.random.randint(100, 10000, len(timestamps))}, index=timestamps)
    player_count_df = pd.DataFrame({'cs2_players': np.random.randint(800000, 1500000, len(timestamps))}, index=timestamps)
    
    df = merge_external_factors(df, liquidity_df, player_count_df)
    
    # Prepare features and target
    target_col = 'spot_price'
    drop_cols = ['category', 'item_name', 'marketplace_provider', target_col]
    X = df.drop(columns=drop_cols, errors='ignore')
    # Retain numeric only for sklearn
    X = X.select_dtypes(include=[np.number]) 
    y = df[target_col]
    
    horizons = [8, 16] # Testing specific horizons to minimize compute during execution
    
    print("3. Validation & Backtesting...")
    benchmark = backtest_pipeline(X, y, horizons=horizons, n_splits=5)
    print("Benchmark Results (MAPE):")
    for algo, res in benchmark.items():
        print(f"  {algo.upper()}:")
        for h, metrics in res.items():
            print(f"    Horizon {h}d -> MAPE: {metrics['mean_mape']:.4f}, RMSE: {metrics['mean_rmse']:.2f}")

    print("4. Model Training & Feature Importance...")
    trainer = MultiHorizonTrainer(horizons=horizons)
    trainer.train_models(X, y)
    print("Top 3 Features for LGBM (8d horizon):")
    print(trainer.get_importance('lightgbm', 8).head(3))

    print("5. Inference & Self-Rating...")
    inference_engine = QuantileInferenceEngine(algo_name='lightgbm')
    inference_engine.fit_horizon(X, y, horizon=8)
    
    # Infer on latest timestamp
    latest_X = X.iloc[[-1]]
    prediction = inference_engine.predict_with_confidence(latest_X, horizon=8)
    print("\nLatest Prediction (8-day horizon):")
    print(prediction)

if __name__ == "__main__":
    main()