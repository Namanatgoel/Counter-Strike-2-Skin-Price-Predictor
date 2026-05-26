import pandas as pd
import numpy as np

def _compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=price.index).rolling(window=window).mean()
    avg_loss = pd.Series(loss, index=price.index).rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=price.index)

def _compute_macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _compute_bollinger_bands(price: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple:
    sma = price.rolling(window=window).mean()
    std = price.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def build_internal_features(df: pd.DataFrame) -> pd.DataFrame:
    def compute_metrics(group):
        group = group.sort_index()
        price = group['spot_price']
        
        group['sma_3'] = price.rolling(window=3).mean()
        group['sma_7'] = price.rolling(window=7).mean()
        group['sma_14'] = price.rolling(window=14).mean()
        
        group['mom_3'] = price.pct_change(periods=3)
        group['mom_7'] = price.pct_change(periods=7)
        
        daily_returns = price.pct_change(periods=1)
        group['volatility_14'] = daily_returns.rolling(window=14).std()
        
        price_diff = price.diff()
        direction = np.sign(price_diff).replace(0, method='ffill').fillna(0)
        trend_blocks = (direction != direction.shift()).cumsum()
        group['trend_duration'] = direction.groupby(trend_blocks).cumsum()
        
        group['rsi_14'] = _compute_rsi(price, window=14)
        macd, macd_signal, macd_histogram = _compute_macd(price, fast=12, slow=26, signal=9)
        group['macd'] = macd
        group['macd_signal'] = macd_signal
        group['macd_histogram'] = macd_histogram
        
        bb_upper, bb_middle, bb_lower = _compute_bollinger_bands(price, window=20, num_std=2.0)
        group['bb_upper'] = bb_upper
        group['bb_middle'] = bb_middle
        group['bb_lower'] = bb_lower
        group['bb_position'] = (price - bb_lower) / (bb_upper - bb_lower + 1e-8)
        
        return group

    engineered_df = df.groupby(['item_name', 'marketplace_provider'], group_keys=False).apply(compute_metrics)
    engineered_df.dropna(inplace=True)
    return engineered_df

def merge_external_factors(df: pd.DataFrame, liquidity_df: pd.DataFrame, player_count_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.join(liquidity_df, how='left')
    merged = merged.join(player_count_df, how='left')
    merged = merged.ffill(limit=2).fillna(merged.median(numeric_only=True))
    return merged