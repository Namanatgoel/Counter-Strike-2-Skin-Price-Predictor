import pandas as pd
import numpy as np

def build_internal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates SMA, Momentum, Volatility, and Trend Duration grouped by item and provider."""
    
    def compute_metrics(group):
        group = group.sort_index()
        price = group['spot_price']
        
        # 1. Rolling Averages (SMA)
        group['sma_3'] = price.rolling(window=3).mean()
        group['sma_7'] = price.rolling(window=7).mean()
        group['sma_14'] = price.rolling(window=14).mean()
        
        # 2. Momentum (% change)
        group['mom_3'] = price.pct_change(periods=3)
        group['mom_7'] = price.pct_change(periods=7)
        
        # 3. Volatility (14-day rolling std of daily returns)
        daily_returns = price.pct_change(periods=1)
        group['volatility_14'] = daily_returns.rolling(window=14).std()
        
        # 4. Trend Duration
        price_diff = price.diff()
        direction = np.sign(price_diff)
        direction = direction.replace(0, method='ffill').fillna(0) # Handle flat days by keeping previous trend
        
        # Identify trend shifts to create blocks
        trend_blocks = (direction != direction.shift()).cumsum()
        # Cumulative sum of direction within each block
        group['trend_duration'] = direction.groupby(trend_blocks).cumsum()
        
        return group

    # Apply feature engineering per distinct item and marketplace
    engineered_df = df.groupby(['item_name', 'marketplace_provider'], group_keys=False).apply(compute_metrics)
    engineered_df.dropna(inplace=True)
    return engineered_df

def merge_external_factors(df: pd.DataFrame, liquidity_df: pd.DataFrame, player_count_df: pd.DataFrame) -> pd.DataFrame:
    """Left joins external factors onto the main dataframe via timestamp index."""
    merged = df.join(liquidity_df, how='left')
    merged = merged.join(player_count_df, how='left')
    # Forward fill missing external data up to 2 days, then fill with median to maintain structure
    merged = merged.ffill(limit=2).fillna(merged.median(numeric_only=True))
    return merged