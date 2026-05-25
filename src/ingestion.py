import pandas as pd
import numpy as np
import ujson
from multiprocessing import Pool, cpu_count
import os

def _parse_category(args):
    """Worker function to parse a single category dictionary."""
    category_name, category_data = args
    records = []
    
    for item_name, providers in category_data.items():
        for provider, time_series in providers.items():
            for ts, price in time_series.items():
                records.append({
                    'timestamp': pd.to_datetime(int(ts), unit='s'),
                    'category': category_name,
                    'item_name': item_name,
                    'marketplace_provider': provider,
                    'spot_price': float(price)
                })
    return records

def load_and_flatten_json(file_path: str) -> pd.DataFrame:
    """Parses nested JSON and flattens it using multiprocessing."""
    with open(file_path, 'r') as f:
        data = ujson.load(f)
        
    category_items = list(data.items())
    
    with Pool(processes=min(cpu_count(), len(category_items))) as pool:
        results = pool.map(_parse_category, category_items)
        
    flat_records = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flat_records)
    
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples to daily frequency, forward fills max 3 days, drops remaining NaNs."""
    def process_group(group):
        # Resample to daily frequency keeping the last price of the day
        resampled = group.resample('D').last()
        # Forward fill up to 3 days
        resampled['spot_price'] = resampled['spot_price'].ffill(limit=3)
        # Drop gaps larger than 3 days
        return resampled.dropna(subset=['spot_price'])

    # Group by unique asset/market combination
    cleaned_df = df.groupby(['category', 'item_name', 'marketplace_provider']).apply(process_group)
    cleaned_df = cleaned_df.drop(columns=['category', 'item_name', 'marketplace_provider']).reset_index()
    cleaned_df.set_index('timestamp', inplace=True)
    cleaned_df.sort_index(inplace=True)
    
    return cleaned_df