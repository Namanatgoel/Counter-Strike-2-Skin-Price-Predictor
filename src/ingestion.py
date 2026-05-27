"""
ingestion.py — Data Loading and Preprocessing Module
=====================================================

Purpose
-------
This module is the **entry point** for all raw data in the CS2 skin price
prediction pipeline. It is responsible for two distinct tasks:

1. **Loading & Flattening**: Reading deeply nested JSON files (category →
   item → marketplace → timestamp → price) into a flat, tabular
   pandas DataFrame that every downstream module can easily consume.

2. **Resampling & Cleaning**: Converting the raw, irregularly-spaced tick
   data into a consistent daily time-series, handling missing days by
   forward-filling up to a sensible gap limit, and removing any
   permanently illiquid (data-sparse) windows.

Design Rationale
----------------
**Why ujson instead of the standard `json` library?**
    The training dataset is ~110 MB of deeply nested JSON. Python's
    built-in `json.load()` is a pure-Python parser and is noticeably
    slow on files this large. `ujson` (Ultra JSON) is a C extension that
    parses the same content 3–5× faster with no API changes.

**Why multiprocessing instead of a simple nested loop?**
    The JSON is structured at the top level as a dictionary of
    *categories* (e.g., "knife", "agent", "rifle"). Each category is
    independently parseable — no shared state is needed. This makes the
    problem embarrassingly parallel. Python's `multiprocessing.Pool`
    spawns worker processes (bypassing the Global Interpreter Lock) that
    each handle one category simultaneously, cutting load time roughly
    in half on a dual-core machine and proportionally more on machines
    with higher core counts.

**Why forward-fill with a limit of 3 days?**
    CS2 marketplace data has natural gaps: items may not trade every
    single day. Short gaps (1–3 days) almost certainly represent
    "no trades today" rather than "the item disappeared from the market",
    so carrying the last known price forward is statistically sound.
    Gaps longer than 3 days, however, suggest the item was genuinely
    illiquid during that window. Filling those would distort momentum
    and volatility features by artificially flattening them. Those rows
    are therefore dropped entirely.

Dependencies
------------
    pandas       — DataFrame construction and time-series resampling
    numpy        — Numeric type conversions (not heavily used here)
    ujson        — Fast C-level JSON parsing
    multiprocessing — Parallel category parsing
    os           — cpu_count() for dynamic worker sizing
"""

import pandas as pd
import numpy as np
import ujson
from multiprocessing import Pool, cpu_count
import os


# ---------------------------------------------------------------------------
# Private worker function — runs inside each spawned worker process
# ---------------------------------------------------------------------------

def _parse_category(args: tuple) -> list:
    """
    Worker function executed in a child process to parse one top-level
    category of the JSON hierarchy.

    This function is intentionally kept at module scope (not nested inside
    another function) because Python's multiprocessing on Windows uses
    'spawn' semantics — it cannot pickle closures or inner functions. By
    keeping it at the top level, it is importable and picklable by all
    platforms.

    Parameters
    ----------
    args : tuple
        A 2-tuple of (category_name: str, category_data: dict).
        `category_data` has the structure:
            { item_name: { provider: { unix_timestamp_str: price_float } } }

    Returns
    -------
    list of dict
        Each dict represents one price tick with the keys:
            - 'timestamp'            : pd.Timestamp (UTC, converted from Unix seconds)
            - 'category'             : str  (e.g. "knife", "agent")
            - 'item_name'            : str  (e.g. "AK-47 | Redline (Field-Tested)")
            - 'marketplace_provider' : str  (e.g. "steam", "skinport")
            - 'spot_price'           : float (price in the denomination stored in JSON)

    Notes
    -----
    Timestamps in the JSON are stored as *string* representations of Unix
    epoch seconds (e.g. "1716998400"). `pd.to_datetime(int(ts), unit='s')`
    converts them correctly to pandas Timestamps in UTC.

    The function intentionally does NOT raise on malformed data — it simply
    skips records that cannot be parsed. This makes ingestion fault-tolerant
    against occasional corrupt marketplace API responses.
    """
    category_name, category_data = args
    records = []

    for item_name, providers in category_data.items():
        for provider, time_series in providers.items():
            for ts, price in time_series.items():
                # Convert Unix epoch string to a proper Timestamp
                records.append({
                    'timestamp': pd.to_datetime(int(ts), unit='s'),
                    'category': category_name,
                    'item_name': item_name,
                    'marketplace_provider': provider,
                    'spot_price': float(price)   # ensure numeric, not string
                })

    return records


# ---------------------------------------------------------------------------
# Public API — called from main.py
# ---------------------------------------------------------------------------

def load_and_flatten_json(file_path: str) -> pd.DataFrame:
    """
    Loads a nested JSON file from disk and returns a flat, chronologically
    sorted DataFrame of price ticks.

    The JSON schema expected by this function is:
    ::

        {
          "<category>": {
            "<item_name>": {
              "<marketplace_provider>": {
                "<unix_timestamp_as_string>": <price_as_number>
              }
            }
          }
        }

    The function achieves parallelism by splitting the top-level category
    dictionary into individual (name, data) pairs and distributing them
    across a `multiprocessing.Pool`.  The worker count is capped at the
    actual number of categories so we never spawn idle workers.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the JSON file on disk.
        Example: 'data/model_training_skin.json'

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (index), category, item_name,
                 marketplace_provider, spot_price
        The DataFrame is indexed by timestamp and sorted ascending.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    ujson.JSONDecodeError
        If the file is not valid JSON.

    Example
    -------
    >>> df = load_and_flatten_json('data/model_training_skin.json')
    >>> df.head()
                         category         item_name marketplace_provider  spot_price
    timestamp
    2024-01-01 00:00:00     knife  Karambit | Fade           steam         1250.00
    """
    # Read the entire file into memory at once — ujson is fast enough
    with open(file_path, 'r') as f:
        data = ujson.load(f)

    # Convert top-level dict to a list of (name, dict) pairs for map()
    category_items = list(data.items())

    # Spawn worker processes — use at most as many workers as categories
    # to avoid idle processes. Also respect the machine's CPU count.
    n_workers = min(cpu_count(), len(category_items))
    with Pool(processes=n_workers) as pool:
        # pool.map distributes one (name, dict) pair per worker call
        results = pool.map(_parse_category, category_items)

    # Flatten the list-of-lists returned by each worker into one flat list
    flat_records = [item for sublist in results for item in sublist]

    # Build DataFrame and set timestamp as the index for time-series operations
    df = pd.DataFrame(flat_records)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)   # chronological order required downstream

    return df


def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts an irregularly-timed tick DataFrame into a clean daily
    time-series, one row per (item, provider, day).

    Processing steps applied to each (category, item_name, provider) group:

    1. **Resample to daily ('D') frequency** — keeps only the *last* price
       seen on each calendar day. This removes intra-day noise while
       preserving end-of-day settlement prices.

    2. **Forward-fill gaps ≤ 3 days** — propagates the most recent observed
       price into subsequent days where no trade occurred, up to a maximum
       of 3 consecutive missing days. This prevents artificial NaN gaps
       from breaking rolling-window calculations later.

    3. **Drop remaining NaN rows** — any day that is still NaN after the
       3-day fill window is removed. This covers extended illiquid periods
       (items not traded for 4+ consecutive days) where carrying the price
       forward would be statistically misleading.

    Parameters
    ----------
    df : pd.DataFrame
        Raw tick DataFrame as returned by `load_and_flatten_json`.
        Must have a DatetimeIndex and columns:
            category, item_name, marketplace_provider, spot_price

    Returns
    -------
    pd.DataFrame
        Daily-frequency DataFrame with the same schema.
        The index is a DatetimeIndex at daily granularity, sorted ascending.

    Notes
    -----
    `groupby(...).apply(process_group)` creates a MultiIndex after the
    apply call. The `reset_index()` call inside the function collapses it
    back to a flat structure with timestamp as the sole index.

    The inner `process_group` function is defined locally because it
    uses only standard pandas and does not need to be picklable (it runs
    in the main process, not a worker process).

    Example
    -------
    >>> df_clean = clean_and_resample(df_raw)
    >>> df_clean.index.freq  # should be None (variable items) but daily-spaced
    """

    def process_group(group: pd.DataFrame) -> pd.DataFrame:
        """
        Applies resampling and filling to a single (item, provider) slice.

        The group is already filtered to one specific item on one specific
        marketplace. We sort it by time, resample to daily granularity
        (keeping the day's last price), then apply the 3-day forward fill.
        """
        group = group.sort_index()

        # 'D' = calendar day. `.last()` keeps the final price in each day's
        # window, acting like an end-of-day settlement price.
        resampled = group.resample('D').last()

        # Forward-fill: carry the last known price forward by up to 3 days.
        # The `limit=3` parameter is the key guard against over-filling.
        resampled['spot_price'] = resampled['spot_price'].ffill(limit=3)

        # Remove any day that remains NaN (gap > 3 days or start of series)
        return resampled.dropna(subset=['spot_price'])

    # Apply the cleaning function to every (category, item, provider) group.
    # `group_keys=False` prevents pandas from prepending the group keys to
    # the index, which would create an unwanted MultiIndex.
    cleaned_df = df.groupby(
        ['category', 'item_name', 'marketplace_provider']
    ).apply(process_group)

    # After groupby.apply, the group keys are in the MultiIndex levels.
    # drop() removes the redundant category/item/provider columns that
    # were folded into the MultiIndex, then reset_index() reconstructs
    # a clean flat structure where `timestamp` is the sole index.
    cleaned_df = cleaned_df.drop(
        columns=['category', 'item_name', 'marketplace_provider']
    ).reset_index()
    cleaned_df.set_index('timestamp', inplace=True)
    cleaned_df.sort_index(inplace=True)

    return cleaned_df
