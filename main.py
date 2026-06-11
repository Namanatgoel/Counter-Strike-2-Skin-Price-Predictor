"""
main.py — Master Orchestration Script
======================================

This is the single entry point for the entire CS2 Skin Price Prediction
pipeline. Running `python main.py` from the project root directory executes
every stage in the correct order:

    1. Data Ingestion     → Load and flatten JSON price data
    2. Feature Engineering → Compute technical indicators
    3. External Factors   → Merge market liquidity and player count data
    4. Hyperparameter Tuning → Bayesian search for optimal LightGBM config
    5. Model Training     → Train all three algorithms for the 8-day horizon
    6. Quantile Inference → Train probabilistic (P5/P50/P95) models
    7. Test Set Evaluation → Load test data, predict, compute metrics
    8. Visualization      → Generate 3 interactive HTML charts

Pipeline Architecture
---------------------
This script follows the "director" pattern: it imports specialized modules
from the `src/` directory and orchestrates them in sequence, passing data
between stages as plain pandas DataFrames. No module in `src/` depends on
any other `src/` module — all wiring happens here in main.py.

Data Flow Diagram
-----------------
JSON File
    ↓  (ingestion.py: load_and_flatten_json)
Raw Tick DataFrame
    ↓  (ingestion.py: clean_and_resample)
Daily Price DataFrame
    ↓  (features.py: build_internal_features)
Feature-Engineered DataFrame
    ↓  (features.py: merge_external_factors)
Full Feature Matrix (X)
    ↓  (this file: compute_targets)
Target Returns Dictionary (y_dict)
    ↓
┌─────────────────────────────────────────┐
│  tuning.py: HyperparameterOptimizer    │ → best_params (dict)
│  training.py: MultiHorizonTrainer      │ → trained models
│  inference.py: QuantileInferenceEngine │ → probabilistic predictions
└─────────────────────────────────────────┘
    ↓
Test Set Metrics (RMSE, MAPE, Directional Accuracy)
    ↓
  visualization.py: 3 × HTML charts saved to output/

Why 8-Day Horizon Only?
-----------------------
The training dataset spans approximately 31 days. Feature engineering
requires a 20-day Bollinger Band warm-up period, leaving ~11 usable rows
per item. A 16-day forecast horizon would require shifting the target
16 rows into the future — leaving only -5 rows of overlap. This makes
16-day forecasting mathematically impossible with this data length.

8-day horizon is the maximum practical horizon given 31 days of data
with 20-day feature warm-up. When more historical data becomes available
(e.g., 90+ days), horizons up to 16 days become viable.

Horizon Formula:
    Usable rows = total_days - warmup_days - horizon
    31 - 20 - 8 = 3 rows → just enough to be meaningful
    31 - 20 - 16 = -5 rows → impossible

Targets: Percentage Returns vs Absolute Prices
----------------------------------------------
The model predicts *percentage returns* (e.g., +0.05 = +5%) rather than
absolute prices (e.g., $12.50). This design choice has two benefits:

1. **Scale invariance**: A $5 item and a $500 item trained together without
   the model being confused by the 100× price difference.

2. **Stationarity**: Price levels are non-stationary (they drift up/down over
   time), but returns are approximately stationary. Stationary data is
   easier for tree models to learn from.

The formula is: return[h] = price(t + h) / price(t) - 1

At inference time, the return is converted back to an absolute price:
    predicted_price = current_price × (1 + predicted_return)

Randomized External Factors
----------------------------
In the current implementation, `liquidity_vol` and `cs2_players` are
generated with `np.random.randint()` as placeholders. In a production
deployment, these would be fetched from real data sources:
- Steam API: historical concurrent player counts
- Marketplace APIs: aggregate trading volume per day

The framework is designed so that replacing the random data with real
data requires changing only 3 lines in `prepare_pipeline()`.
"""

import pandas as pd
import numpy as np
import os
import warnings
import ujson

# Import all pipeline modules from the src/ package
from src.ingestion import clean_and_resample
from src.features import build_internal_features, merge_external_factors, join_static_metadata
from src.metadata import fetch_api_skins, extract_api_features
from src.training import MultiHorizonTrainer
from src.inference import QuantileInferenceEngine
from src.tuning import HyperparameterOptimizer
from src.visualization import plot_forecast_intervals, plot_shap_summary, plot_volatility_regimes

# Suppress deprecation and future warnings from third-party libraries
# to keep the console output clean during training
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_training_json(file_path: str) -> pd.DataFrame:
    """
    Loads the current item-level JSON file and flattens it to tick data.

    The new schema is keyed by item name at the top level. Each item stores
    metadata fields such as `cat`, `variant`, `base`, and `rkey`, plus a
    nested `providers` dictionary containing timestamped prices.
    """
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as file_handle:
        data = ujson.load(file_handle)

    records = []
    if not data:
        return pd.DataFrame(columns=['timestamp', 'category', 'item_name', 'marketplace_provider', 'spot_price'])

    sample_value = next(iter(data.values()))
    if isinstance(sample_value, dict) and 'providers' in sample_value:
        for item_name, item_data in data.items():
            item_metadata = {
                key: value for key, value in item_data.items()
                if key != 'providers' and not isinstance(value, dict)
            }
            category = item_metadata.pop('cat', item_metadata.pop('category', None))
            providers = item_data.get('providers', {})

            for provider, time_series in providers.items():
                for ts, price in time_series.items():
                    record = {
                        'timestamp': pd.to_datetime(int(ts), unit='s'),
                        'category': category,
                        'item_name': item_name,
                        'marketplace_provider': provider,
                        'spot_price': float(price),
                    }
                    record.update(item_metadata)
                    records.append(record)
    else:
        for category_name, category_data in data.items():
            for item_name, providers in category_data.items():
                for provider, time_series in providers.items():
                    for ts, price in time_series.items():
                        records.append({
                            'timestamp': pd.to_datetime(int(ts), unit='s'),
                            'category': category_name,
                            'item_name': item_name,
                            'marketplace_provider': provider,
                            'spot_price': float(price),
                        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def temporal_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
    """
    Splits a time-indexed frame into chronological train and test sets.

    The split is done on unique timestamps, not on individual rows, so all
    items from the same day stay together in the same partition.
    """
    if df.empty:
        raise ValueError('Cannot split an empty DataFrame.')

    ordered_df = df.sort_index()
    unique_timestamps = ordered_df.index.unique().sort_values()
    if len(unique_timestamps) < 2:
        raise ValueError('Need at least two unique timestamps to create a train/test split.')

    split_position = int(len(unique_timestamps) * train_ratio)
    split_position = max(1, min(len(unique_timestamps) - 1, split_position))
    split_timestamp = unique_timestamps[split_position]

    train_df = ordered_df[ordered_df.index < split_timestamp].copy()
    test_df = ordered_df[ordered_df.index >= split_timestamp].copy()
    return train_df, test_df


def build_model_features(df: pd.DataFrame) -> tuple:
    """
    Removes non-feature columns and converts string columns to categorical.
    """
    features = df.drop(columns=['item_name', 'spot_price'], errors='ignore').copy()
    categorical_columns = features.select_dtypes(include=['object']).columns.tolist()

    for column in categorical_columns:
        features[column] = features[column].astype(str).astype('category')

    return features, categorical_columns

def compute_targets(df: pd.DataFrame, horizons: list) -> dict:
    """
    Computes the prediction targets for each forecast horizon.

    The target for horizon h at time t is the percentage price return
    observed h days after t:

        y[t, h] = (price[t + h] / price[t]) - 1

    Where:
        - price[t]     is the current spot price
        - price[t + h] is the spot price h days in the future (from the dataset)

    This is computed using `groupby` + `shift(-h)` to correctly handle
    multiple items in the same DataFrame. Without groupby, `shift(-h)` would
    bleed the end of one item's series into the beginning of the next item's
    series (the "boundary problem").

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame. Must contain columns:
        'item_name', 'marketplace_provider', 'spot_price'.
        Index must be a DatetimeIndex.

    horizons : list of int
        Forecast horizons in days. For each h, a separate target Series
        is computed.

    Returns
    -------
    dict
        Keys are horizon integers; values are pd.Series of returns.
        Example: {8: pd.Series([0.02, -0.01, 0.00, ...])}

        The last `h` rows for each item will be NaN because the future
        price is not yet known in the dataset. These NaN rows are
        automatically excluded during training.

    Example
    -------
    >>> y_dict = compute_targets(df, [8])
    >>> y_dict[8].describe()
    count    1234.000
    mean        0.003
    std         0.028
    ...
    """
    y_dict = {}
    for h in horizons:
        # shift(-h) moves each row's value h positions earlier in the index.
        # For a group that ends on day 30, shift(-8) gives the day-30 row
        # the day-30+8=38 price. Since day 38 doesn't exist, it becomes NaN.
        target_price = df.groupby(
            ['item_name', 'marketplace_provider']
        )['spot_price'].shift(-h)

        # Compute percentage return: how much does the price change?
        y_dict[h] = target_price / df['spot_price'] - 1

    return y_dict


def prepare_feature_frame(file_path: str, metadata_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Executes the full data preparation pipeline for one JSON dataset.

    The returned frame is later split chronologically into train and test
    partitions so the holdout period remains a true future-facing evaluation.

    Steps executed:
    1. Load the item-level JSON and flatten it to tabular format
    2. Resample to daily frequency and fill short gaps (ingestion.py)
    3. Compute all technical indicator features (features.py)
    4. Simulate and merge external macro factors (features.py)
    5. Join static API metadata (features.py)

    Parameters
    ----------
    file_path : str
        Path to the input JSON file.

    metadata_df : pd.DataFrame, optional
        Static metadata from the CSGO API (rarity, float caps, etc.).
        Indexed by item_name. If None, metadata join is skipped.

    Returns
    -------
    pd.DataFrame
        Full feature-engineered DataFrame, ready for temporal splitting.

    Notes on External Factor Simulation
    ------------------------------------
    `liquidity_df` and `player_count_df` are generated here with random
    integers within realistic ranges:
    - liquidity_vol: 100 to 10,000 (units represent trading volume)
    - cs2_players: 800,000 to 1,500,000 (Steam concurrent player count)

    Both DataFrames are indexed on the unique timestamps present in the
    data, matching the granularity of the price data exactly.

    For production, replace these two lines with real API data fetches.
    """
    df = load_training_json(file_path)
    df = clean_and_resample(df)

    print("Building features...")
    df = build_internal_features(df)

    # Step 4: Create external factor DataFrames aligned to the data's timestamps
    timestamps = df.index.unique()

    # --- PLACEHOLDER: Replace with real liquidity data from marketplace APIs ---
    liquidity_df = pd.DataFrame(
        {'liquidity_vol': np.random.randint(100, 10000, len(timestamps))},
        index=timestamps
    )

    # --- PLACEHOLDER: Replace with real Steam concurrent player count data ---
    player_count_df = pd.DataFrame(
        {'cs2_players': np.random.randint(800000, 1500000, len(timestamps))},
        index=timestamps
    )

    # Merge external factors into the feature matrix
    df = merge_external_factors(df, liquidity_df, player_count_df)

    # Step 5: Join static API metadata if available
    if metadata_df is not None:
        print("Joining API metadata...")
        df = join_static_metadata(df, metadata_df)

    return df


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """
    Master orchestration function — executes the complete ML pipeline.

    Called when running `python main.py` from the project root.
    See module docstring above for a detailed description of each stage.
    """

    # Ensure output directory exists for saving visualizations
    os.makedirs('output', exist_ok=True)

    # -----------------------------------------------------------------------
    # STAGE 1: Fetch API Metadata
    # -----------------------------------------------------------------------
    print("\n--- Fetching API Metadata ---")
    raw_api_df = fetch_api_skins()
    metadata_df = extract_api_features(raw_api_df)
    print(f"  Extracted metadata for {len(metadata_df)} item name variants")

    # -----------------------------------------------------------------------
    # STAGE 2: Choose Forecast Horizons
    # -----------------------------------------------------------------------
    # See module docstring for explanation of why only [8] is used.
    # To experiment with other horizons, ensure you have 30+ extra days of data
    # beyond the warm-up period (20 days for Bollinger Bands).
    horizons = [8]

    # -----------------------------------------------------------------------
    # STAGE 3: Prepare Feature Frame and Split Chronologically
    # -----------------------------------------------------------------------
    # The JSON file now contains the full timeline in one file, so we build the
    # feature frame once and then create an 80/20 time-based holdout split.
    feature_frame = prepare_feature_frame(
        'data/model_training.json', metadata_df=metadata_df
    )
    train_frame, test_frame = temporal_train_test_split(feature_frame, train_ratio=0.8)
    print(
        f"  Temporal split: {len(train_frame):,} training rows / {len(test_frame):,} test rows"
    )

    y_train_dict = compute_targets(train_frame, horizons)
    y_test_dict = compute_targets(test_frame, horizons)

    X_train = train_frame.copy()
    X_test = test_frame.copy()

    # -----------------------------------------------------------------------
    # STAGE 4: Hyperparameter Optimization
    # -----------------------------------------------------------------------
    # Run Bayesian optimization to find the best LightGBM configuration.
    # `groups=X_train['item_name']` passes item names to GroupKFold so
    # that different rows of the same item always land in the same fold.
    print("\n--- Hyperparameter Optimization (8-Day Horizon) ---")
    optimizer = HyperparameterOptimizer(n_trials=20, horizon=8)

    # Separate the item identifier and price columns from features
    # (the model should not see 'item_name' or 'spot_price' as input features)
    groups = X_train['item_name']
    X_train_features, cat_cols = build_model_features(X_train)

    best_params = optimizer.optimize(X_train_features, y_train_dict[8], groups=groups)

    print("Best Hyperparameters Found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Train the "best model" on full training data with optimal params
    # (used for SHAP visualization, not for the final inference predictions)
    best_model = optimizer.get_best_model(X_train_features, y_train_dict[8])

    # -----------------------------------------------------------------------\n    # STAGE 5: Multi-Algorithm Training
    # -----------------------------------------------------------------------
    # Train all three algorithms (HistGBM, XGBoost, LightGBM) for the
    # 8-day horizon using their default hyperparameters.
    # This populates trainer.models[algo][8] for comparison.
    print("\n--- Training Multi-Algorithm Models ---")
    trainer = MultiHorizonTrainer(horizons=horizons)
    trainer.train_models(X_train_features, y_train_dict)

    # -----------------------------------------------------------------------
    # STAGE 6: Quantile Inference Training
    # -----------------------------------------------------------------------
    # Train the probabilistic (P5/P50/P95) models using LightGBM quantile
    # regression. This is separate from the point-estimate models above.
    inference_engine = QuantileInferenceEngine(algo_name='lightgbm')
    inference_engine.fit_horizon(X_train_features, y_train_dict[8], horizon=8)

    # -----------------------------------------------------------------------
    # STAGE 7: Prepare Test Data
    # -----------------------------------------------------------------------
    # The test partition uses the same feature pipeline but a later time slice.
    print("\n--- Processing Test Set ---")
    X_test_features, _ = build_model_features(X_test)

    # Align categorical dtypes to the training schema.
    for col in cat_cols:
        if col in X_test_features.columns:
            X_test_features[col] = X_test_features[col].astype('category')

    # -----------------------------------------------------------------------
    # STAGE 7: Inference & Evaluation on Test Data
    # -----------------------------------------------------------------------
    # Generate probabilistic predictions on the test set and compute
    # three performance metrics:
    #   1. MAPE  — average percentage error
    #   2. RMSE  — average dollar error
    #   3. Directional Accuracy — did we predict up/down correctly?
    print("\n--- Evaluation (8-Day Horizon) ---")

    # Create validity mask: exclude test rows where the 8-day future is unknown
    y_target_8d_ret = y_test_dict[8]
    valid_idx = y_target_8d_ret.notna()

    # Slice to only valid rows for inference
    X_test_valid   = X_test_features[valid_idx]
    y_test_valid   = y_target_8d_ret[valid_idx]
    prices_valid   = X_test.loc[valid_idx, 'spot_price']   # current prices (for return→price conversion)
    items_valid    = X_test.loc[valid_idx, 'item_name']    # item names (for selecting sample item)

    # Generate predictions: returns P5, P50, P95, and Confidence Score
    predictions = inference_engine.predict_with_confidence(
        X_test_valid, prices_valid, horizon=8
    )

    # Reconstruct true future prices from the target returns:
    #   true_price = current_price × (1 + true_return)
    true_prices = prices_valid * (1 + y_test_valid)

    # --- Metric 1: MAPE (Mean Absolute Percentage Error) ---
    # Average of |true - predicted| / true, as a percentage
    mape = np.mean(
        np.abs((true_prices - predictions['Prediction_P50']) / true_prices)
    ) * 100

    # --- Metric 2: RMSE (Root Mean Squared Error) ---
    # Square root of average squared errors — penalizes large errors
    rmse = np.sqrt(
        np.mean((true_prices - predictions['Prediction_P50'])**2)
    )

    # --- Metric 3: Directional Accuracy ---
    # Did the model correctly predict whether the price would go up or down?
    # np.sign: +1 for positive, -1 for negative, 0 for zero
    actual_dir = np.sign(y_test_valid)
    pred_dir   = np.sign(
        (predictions['Prediction_P50'] - prices_valid) / prices_valid
    )
    dir_acc = np.mean(actual_dir == pred_dir) * 100

    # Display metrics
    print(f"  Test RMSE:               ${rmse:.4f}")
    print(f"  Test MAPE:               {mape:.2f}%")
    print(f"  Directional Accuracy:    {dir_acc:.2f}%")

    # -----------------------------------------------------------------------
    # OUTPUT: Save predictions to Excel (or CSV fallback)
    # -----------------------------------------------------------------------
    # Construct a tidy DataFrame with one row per prediction containing:
    # - item name, current date, predicted future date, current price
    # - true future price (from test set), P5/P50/P95 and confidence
    try:
        output_df = predictions.copy()
        # Attach identifying columns from the test set (aligned to valid_idx)
        output_df['item_name'] = items_valid.values
        output_df['current_date'] = output_df.index
        # Predictions are for t + horizon days
        output_df['predicted_date'] = output_df.index + pd.Timedelta(days=8)
        output_df['current_price'] = prices_valid.values
        output_df['true_price'] = true_prices.values

        # Reorder columns for readability
        cols = [
            'item_name', 'current_date', 'predicted_date', 'current_price',
            'true_price', 'P5', 'Prediction_P50', 'P95', 'Confidence_Score'
        ]
        # Some columns may be missing in edge cases; filter to available ones
        cols = [c for c in cols if c in output_df.columns]
        output_df = output_df[cols]

        os.makedirs('output', exist_ok=True)
        xlsx_path = os.path.join('output', 'predictions.xlsx')
        try:
            output_df.to_excel(xlsx_path, index=False)
            print(f"  Excel predictions saved to {xlsx_path}")
        except Exception as e:
            # If Excel writer not available, fall back to CSV
            csv_path = os.path.join('output', 'predictions.csv')
            output_df.to_csv(csv_path, index=False)
            print(f"  Could not write Excel file ({e}); saved CSV to {csv_path}")
    except Exception as e:
        print(f"  Warning: failed to assemble predictions output: {e}")

    # -----------------------------------------------------------------------
    # STAGE 8: Visualization
    # -----------------------------------------------------------------------
    # Select one random test item to visualize, then generate all 3 charts.
    print("\n--- Generating Visualizations ---")

    # Pick one random item from the test set (random_state=42 for reproducibility)
    sample_item = items_valid.sample(1, random_state=42).iloc[0]
    print(f"  Plotting forecast for: {sample_item}")

    item_mask = items_valid == sample_item

    # Get actual price history for the sampled item from the test set
    sample_actual = test_frame[test_frame['item_name'] == sample_item][['spot_price']]

    # Get the model's predictions for the sampled item
    sample_pred_full = predictions[item_mask].copy()

    # Shift prediction index forward by 8 days so the prediction line plots
    # at the correct future date (predictions are made "today" but refer to day+8)
    sample_pred_full.index = sample_pred_full.index + pd.Timedelta(days=8)

    # Chart 1: Forecast intervals (actual vs predicted with confidence band)
    plot_forecast_intervals(
        sample_actual,
        sample_pred_full,
        item_name=sample_item,
        output_path='output/forecast_intervals.html'
    )

    # Chart 2: SHAP feature importance summary
    # Uses the best_model (trained with optimal hyperparameters)
    # Only includes rows where the 8-day target is available
    valid_train_idx = y_train_dict[8].notna()
    plot_shap_summary(
        best_model,
        X_train_features[valid_train_idx],
        output_path='output/shap_summary.html'
    )

    # Chart 3: Price and volatility regime visualization for the sample item
    sample_vol = test_frame[test_frame['item_name'] == sample_item][['spot_price', 'volatility_14']]
    plot_volatility_regimes(
        sample_vol,
        output_path='output/volatility_regimes.html'
    )

    print("\n Pipeline Complete.")
    print("  Outputs saved to output/:")
    print("    - forecast_intervals.html  (open in any web browser)")
    print("    - shap_summary.html        (open in any web browser)")
    print("    - volatility_regimes.html  (open in any web browser)")


# ---------------------------------------------------------------------------
# Entry Point Guard
# ---------------------------------------------------------------------------
# This block ensures that `main()` is only called when this file is run
# directly (`python main.py`), NOT when it is imported as a module by
# another script (e.g., in tests or notebooks).
if __name__ == "__main__":
    main()
