"""
features.py — Technical Indicator & Feature Engineering Module
==============================================================

Purpose
-------
This module transforms raw price series into a rich set of numeric
features that machine learning models can learn from. Raw prices alone
are not useful inputs to a gradient boosting model because the model
has no notion of "this price is high relative to recent history" or
"the market has been trending up for 5 days". Feature engineering
translates those intuitions into concrete numbers.

Feature Categories Built Here
------------------------------
1. **Moving Average Ratios (sma_3, sma_7, sma_14)**
   - How far above or below recent average is the current price?
   - Captures mean-reversion tendency common in illiquid markets.

2. **Momentum (mom_3, mom_7)**
   - Percentage price change over 3 and 7 days.
   - Captures directional continuation ("pump") or exhaustion signals.

3. **Volatility (volatility_14)**
   - Rolling 14-day standard deviation of daily returns.
   - CS2 skin prices exhibit volatility clustering: calm periods are
     followed by calm, and turbulent periods by turbulent. This feature
     lets the model detect which regime it is in.

4. **Trend Duration (trend_duration)**
   - How many consecutive days has the price been moving in the same
     direction? A long unbroken uptrend is a classic precursor to a
     "pump-and-dump" reversal.

5. **RSI — Relative Strength Index (rsi_14)**
   - Classic oscillator: values > 70 signal overbought, < 30 oversold.
   - Directly relevant to detecting over-extended pump cycles.

6. **MACD — Moving Average Convergence Divergence (macd, macd_signal, macd_histogram)**
   - Momentum indicator based on the difference between fast and slow
     Exponential Moving Averages.
   - The histogram (MACD minus Signal) is particularly useful for
     detecting early momentum shifts.

7. **Bollinger Bands (bb_upper, bb_lower, bb_middle, bb_position)**
   - Statistical envelope ±2 standard deviations around a 20-day SMA.
   - `bb_position` (0 = at lower band, 1 = at upper band) gives a
     normalized measure of where the price sits within its recent range.

8. **External Factors (liquidity_vol, cs2_players)**
   - Merged from external DataFrames representing market-wide liquidity
     and CS2 concurrent player count. These give the model context
     about macro conditions that affect all skins simultaneously.

Design Rationale
----------------
**Why normalize price-derived features by dividing by current price?**
    A skin worth $5 and one worth $500 are on very different scales.
    If we passed raw SMA values, the model would learn weights that only
    work for a specific price range. By expressing everything as a ratio
    or percentage change, features become scale-invariant — the model
    learns patterns that generalize across cheap and expensive skins.

**Why groupby before computing features?**
    Each (item_name, marketplace_provider) pair has its own independent
    price history. Rolling windows must not bleed across item boundaries.
    `groupby(...).apply()` ensures that a 14-day rolling window for
    "AK-47 | Redline" never incorporates prices from "M4A4 | Howl".

**Why dropna() at the end?**
    Rolling windows with period N require at least N prior observations.
    For a 20-day Bollinger Band, the first 19 rows of each item's history
    will be NaN. Since the model cannot train on NaN inputs, we drop them.
    This is an expected and acceptable loss of the "warm-up" period.

Dependencies
------------
    pandas    — DataFrame operations and groupby
    numpy     — Vectorized arithmetic for gain/loss arrays in RSI
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Private helper functions — pure technical indicator computations
# ---------------------------------------------------------------------------

def _compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """
    Computes the Relative Strength Index (RSI) for a price series.

    RSI measures the speed and magnitude of recent price changes to
    evaluate whether an asset is overbought (RSI > 70) or oversold (RSI < 30).

    Algorithm (Wilder's Smoothing Method)
    -------------------------------------
    1. Compute daily price differences: delta = price - price.shift(1)
    2. Separate into positive gains and negative losses.
    3. Compute rolling mean of gains and rolling mean of losses over `window` days.
    4. RS = avg_gain / avg_loss
    5. RSI = 100 - (100 / (1 + RS))

    Parameters
    ----------
    price : pd.Series
        Daily closing prices for a single item on a single marketplace.
        Must be sorted chronologically.
    window : int, default 14
        Lookback period in days. 14 is the industry-standard default
        popularized by J. Welles Wilder in 1978.

    Returns
    -------
    pd.Series
        RSI values in the range [0, 100]. First `window` values are NaN
        due to the rolling warm-up period.

    Implementation Note
    -------------------
    The `+ 1e-8` in the denominator prevents division-by-zero when
    `avg_loss` is exactly 0 (which happens in a sustained uptrend where
    no down days occurred in the window). Without this guard, RSI would
    return `inf` rather than the correct value of 100.
    """
    delta = price.diff()  # Day-over-day change; first value is NaN

    # numpy.where is used here for vectorized conditional assignment —
    # faster than applying a lambda row by row.
    gain = np.where(delta > 0, delta, 0)   # only positive moves count as gains
    loss = np.where(delta < 0, -delta, 0)  # only negative moves count as losses; flip sign

    # Convert back to Series so we can use .rolling()
    avg_gain = pd.Series(gain, index=price.index).rolling(window=window).mean()
    avg_loss = pd.Series(loss, index=price.index).rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-8)  # Relative Strength ratio; epsilon prevents divide-by-zero
    rsi = 100 - (100 / (1 + rs))       # Normalize to 0–100 scale

    return pd.Series(rsi, index=price.index)


def _compute_macd(
    price: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple:
    """
    Computes the MACD (Moving Average Convergence Divergence) indicator.

    MACD is a trend-following momentum oscillator that reveals changes in the
    strength, direction, and duration of a price trend.

    Components
    ----------
    - MACD Line    : EMA(12) - EMA(26)
                     Positive when the short-term trend is stronger than long-term.
    - Signal Line  : EMA(9) of MACD Line
                     Acts as a trigger; crossovers signal momentum changes.
    - Histogram    : MACD Line - Signal Line
                     Visualizes the distance between MACD and Signal;
                     divergence from zero indicates accelerating momentum.

    Parameters
    ----------
    price  : pd.Series — daily prices for one item/provider pair
    fast   : int — short EMA period (default 12 days)
    slow   : int — long EMA period (default 26 days)
    signal : int — signal line EMA period (default 9 days)

    Returns
    -------
    tuple of (macd_line, signal_line, histogram) — each a pd.Series

    Notes
    -----
    `ewm(span=N, adjust=False)` uses the standard formula where the weight
    on the most recent observation is 2/(N+1). `adjust=False` means we
    use a recursive (not expanding-window) formula, matching the standard
    finance convention.
    """
    # Exponential Moving Averages — recent data weighted more heavily than old
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow              # momentum: positive = short-term strength
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()  # smoothed trigger
    histogram = macd_line - signal_line           # divergence from trigger

    return macd_line, signal_line, histogram


def _compute_bollinger_bands(
    price: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> tuple:
    """
    Computes Bollinger Bands — a statistical price envelope.

    Bollinger Bands place upper and lower bounds at N standard deviations
    above and below a simple moving average. They expand during volatile
    periods and contract during calm ones, providing a dynamic reference
    for "normal" price range.

    Parameters
    ----------
    price   : pd.Series — daily prices
    window  : int — rolling window for SMA and std (default 20 days)
    num_std : float — number of standard deviations for band width (default 2.0)

    Returns
    -------
    tuple of (upper_band, middle_band, lower_band) — each a pd.Series

    Usage Context
    -------------
    The caller further derives `bb_position = (price - lower) / (upper - lower)`,
    which normalizes position within the band to [0, 1].
    A value near 1.0 means the price is at the top of its recent statistical
    range — a common precursor to mean reversion.
    """
    sma = price.rolling(window=window).mean()   # middle band = 20-day SMA
    std = price.rolling(window=window).std()    # rolling standard deviation

    upper = sma + (std * num_std)  # upper band = mean + 2σ
    lower = sma - (std * num_std)  # lower band = mean - 2σ

    return upper, sma, lower


# ---------------------------------------------------------------------------
# Public API — called from main.py
# ---------------------------------------------------------------------------

def build_internal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all technical indicator columns to the DataFrame by computing
    them separately within each (item_name, marketplace_provider) group.

    This is the primary feature-engineering function of the pipeline.
    It calls all private helper functions above and adds the results
    as new columns to the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Clean daily price DataFrame from `clean_and_resample`.
        Must contain: spot_price, item_name, marketplace_provider columns.
        Index must be a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with these new columns:
        - sma_3, sma_7, sma_14      : Moving average ratios (price-normalized)
        - mom_3, mom_7               : 3-day and 7-day momentum (% change)
        - volatility_14              : 14-day rolling volatility (std of daily returns)
        - trend_duration             : Consecutive days in current price direction
        - rsi_14                     : 14-day Relative Strength Index
        - macd, macd_signal, macd_histogram : MACD components (price-normalized)
        - bb_upper, bb_middle, bb_lower    : Bollinger Band distances (price-normalized)
        - bb_position                : Relative position within Bollinger Bands [0,1]

        Rows with NaN values (warm-up period) are dropped via `dropna()`.

    Notes on Price Normalization
    ----------------------------
    Every price-derived feature (SMAs, MACD components, Bollinger Bands)
    is divided by the current `spot_price` and expressed as a ratio or
    percentage deviation. This makes features comparable across items of
    wildly different absolute price levels.

    Example: sma_7 = (7-day SMA / current price) - 1
        → positive means price has risen above its 7-day average
        → negative means price has fallen below its 7-day average
    """

    def compute_metrics(group: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all features for a single (item, provider) slice.

        Called once per group by groupby.apply. The `group` variable here
        is an isolated sub-DataFrame containing only one item's data.
        """
        group = group.sort_index()   # ensure chronological order within group
        price = group['spot_price']  # convenience alias for repeated use

        # ---------------------------------------------------------------
        # 1. Moving Average Ratios
        # ---------------------------------------------------------------
        # Expressed as (MA / price - 1): zero means price = MA, positive
        # means price is above its average (potential sell signal),
        # negative means price is below (potential buy signal).
        group['sma_3']  = price.rolling(window=3).mean()  / price - 1
        group['sma_7']  = price.rolling(window=7).mean()  / price - 1
        group['sma_14'] = price.rolling(window=14).mean() / price - 1

        # ---------------------------------------------------------------
        # 2. Momentum (percentage price change over N days)
        # ---------------------------------------------------------------
        # pct_change(periods=N) computes (price[t] - price[t-N]) / price[t-N]
        # Captures the directional strength of recent moves.
        group['mom_3'] = price.pct_change(periods=3)
        group['mom_7'] = price.pct_change(periods=7)

        # ---------------------------------------------------------------
        # 3. Volatility (rolling 14-day std of daily returns)
        # ---------------------------------------------------------------
        # Daily return = price change from one day to the next as a fraction.
        # Standard deviation of those returns = realized volatility.
        daily_returns = price.pct_change(periods=1)
        group['volatility_14'] = daily_returns.rolling(window=14).std()

        # ---------------------------------------------------------------
        # 4. Trend Duration
        # ---------------------------------------------------------------
        # How many consecutive days has the price been going in the same
        # direction (up or down)?
        #
        # Step-by-step:
        #   a) price_diff = day-over-day price change
        #   b) direction  = +1 (up), -1 (down), or 0 (flat)
        #   c) trend_blocks: a new integer id increments every time
        #      direction changes — groups consecutive same-direction days
        #   d) cumsum within each block gives the duration count
        price_diff  = price.diff()
        direction   = np.sign(price_diff).replace(0, method='ffill').fillna(0)
        trend_blocks = (direction != direction.shift()).cumsum()
        group['trend_duration'] = direction.groupby(trend_blocks).cumsum()

        # ---------------------------------------------------------------
        # 5. RSI — Relative Strength Index
        # ---------------------------------------------------------------
        group['rsi_14'] = _compute_rsi(price, window=14)

        # ---------------------------------------------------------------
        # 6. MACD Components (price-normalized)
        # ---------------------------------------------------------------
        macd, macd_signal, macd_histogram = _compute_macd(price, fast=12, slow=26, signal=9)

        # Divide by price so values are scale-invariant across items
        group['macd']           = macd           / price
        group['macd_signal']    = macd_signal    / price
        group['macd_histogram'] = macd_histogram / price

        # ---------------------------------------------------------------
        # 7. Bollinger Bands (price-normalized)
        # ---------------------------------------------------------------
        bb_upper, bb_middle, bb_lower = _compute_bollinger_bands(price, window=20, num_std=2.0)

        # Express bands as percentage deviation from current price
        group['bb_upper']  = bb_upper  / price - 1   # how far above current price is upper band?
        group['bb_middle'] = bb_middle / price - 1   # how far above/below is the 20-day SMA?
        group['bb_lower']  = bb_lower  / price - 1   # how far below is the lower band?

        # Position within band: 0 = at lower band, 1 = at upper band
        # The 1e-8 epsilon prevents division-by-zero when bands converge (very low volatility)
        group['bb_position'] = (price - bb_lower) / (bb_upper - bb_lower + 1e-8)

        return group

    # Apply the per-group computation across all items and providers.
    # group_keys=False prevents creating a hierarchical index from the group keys.
    engineered_df = df.groupby(
        ['item_name', 'marketplace_provider'],
        group_keys=False
    ).apply(compute_metrics)

    # Drop rows with NaN values that result from the rolling warm-up period.
    # For a 20-day Bollinger Band, the first 19 rows of each item's data
    # will be NaN and must be removed before training.
    engineered_df.dropna(inplace=True)

    return engineered_df


def merge_external_factors(
    df: pd.DataFrame,
    liquidity_df: pd.DataFrame,
    player_count_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges external macro-economic signals into the feature matrix.

    These "external factors" provide market-wide context that affects all
    skins simultaneously but is not captured in any individual item's price
    history:

    - **liquidity_vol**: Total trading volume across all CS2 marketplaces
      on a given day. High liquidity days tend to have tighter bid-ask
      spreads and more reliable price discovery.

    - **cs2_players**: Steam's concurrent player count for CS2. Spikes
      (new updates, tournaments, streamer events) drive demand and can
      temporarily inflate prices across the entire market.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame from `build_internal_features`.
        Index must be a DatetimeIndex at daily frequency.

    liquidity_df : pd.DataFrame
        Single-column DataFrame with column 'liquidity_vol' and a
        DatetimeIndex aligned to the same daily frequency as `df`.
        In production, this would be fetched from a market data API.
        In the current implementation, it is simulated in main.py.

    player_count_df : pd.DataFrame
        Single-column DataFrame with column 'cs2_players' and a
        DatetimeIndex aligned to the same daily frequency as `df`.
        In production, sourced from Steam API historical data.

    Returns
    -------
    pd.DataFrame
        `df` with two additional columns: 'liquidity_vol', 'cs2_players'.
        Missing values (dates not present in external DataFrames) are
        forward-filled by up to 2 days, then filled with the column median
        as a fallback. This ensures no NaN values reach the model.

    Notes on Join Strategy
    ----------------------
    `how='left'` preserves all rows in `df` even if a corresponding date
    is missing from the external DataFrames — we never want to lose item
    data just because a macro feed had a gap. The subsequent fill logic
    handles those gaps gracefully.
    """
    # Left join: keep all rows from df; add external columns where dates match
    merged = df.join(liquidity_df,     how='left')
    merged = merged.join(player_count_df, how='left')

    # Forward-fill external signals up to 2 days (handles weekend/holiday gaps)
    # then fall back to column median for any remaining NaN values
    merged = merged.ffill(limit=2).fillna(merged.median(numeric_only=True))

    return merged


def join_static_metadata(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Left-joins static game metadata (rarity, float caps, collection, etc.)
    onto the time-series price DataFrame.

    Static metadata columns are constant for a given item — they do not
    change over time. The join broadcasts each item's metadata across all
    of that item's chronological rows.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame from `build_internal_features` or
        `merge_external_factors`. Must contain an 'item_name' column.
        Index is a DatetimeIndex.

    metadata_df : pd.DataFrame
        Static metadata DataFrame from `extract_api_features()`.
        Indexed by `item_name`. Columns include:
        'rarity_ordinal', 'is_stattrak', 'is_souvenir', 'min_float',
        'max_float', 'float_range', 'collection', 'weapon_name'.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with metadata columns appended via left join.
        Items not found in the API metadata will have NaN for metadata
        columns, which are filled with defaults:
        - Numeric columns: filled with 0
        - String columns ('collection', 'weapon_name'): filled with 'Unknown'

    Notes on Join Strategy
    ----------------------
    The join is performed on the 'item_name' column (not the index) because
    the main DataFrame's index is a DatetimeIndex. `metadata_df` is indexed
    by item_name, so we use `df['item_name'].map(metadata_df[col])` for
    efficient broadcasting without resetting indices.
    """
    # Join each metadata column individually via .map() on item_name
    # This is more robust than merge() when the main df has a DatetimeIndex
    for col in metadata_df.columns:
        df[col] = df['item_name'].map(metadata_df[col])

    # Fill missing metadata for items not found in the API
    numeric_meta_cols = [
        'rarity_ordinal', 'is_stattrak', 'is_souvenir',
        'min_float', 'max_float', 'float_range'
    ]
    for col in numeric_meta_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    string_meta_cols = ['collection', 'weapon_name']
    for col in string_meta_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Report join coverage
    matched = df['item_name'].isin(metadata_df.index).sum()
    total = len(df)
    unique_matched = df.loc[df['item_name'].isin(metadata_df.index), 'item_name'].nunique()
    unique_total = df['item_name'].nunique()
    print(f"  Metadata join: {unique_matched}/{unique_total} unique items matched "
          f"({matched}/{total} rows)")

    return df

