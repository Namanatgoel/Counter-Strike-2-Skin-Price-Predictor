"""
metadata.py — Static Game Metadata Ingestion from ByMykel/CSGO-API
===================================================================

Fetches and caches skin metadata (rarity, float caps, StatTrak/Souvenir
status, collections) from the ByMykel CSGO-API GitHub repository.

The API provides a comprehensive static dataset of all CS2 skin definitions
including properties that do not change over time (rarity tier, float range,
whether the item can be StatTrak or Souvenir). These are valuable features
for price prediction because they define the fundamental supply characteristics
of each skin.

Caching
-------
The raw JSON response is cached locally at `data/api_skins_cache.json`.
If the cache file exists and is less than 7 days old, data is loaded from
disk without making a network request. This avoids hammering the API on
repeated pipeline runs and ensures offline reproducibility.

Name Matching Strategy
----------------------
The API provides skin names WITHOUT wear condition suffixes:
    API:     "AK-47 | Redline"
    Dataset: "AK-47 | Redline (Field-Tested)"

To join, we strip the wear suffix "(Factory New)", "(Minimal Wear)", etc.
from dataset item names to produce a "base_name" key. Additionally, the API
uses "StatTrak™" prefix while the dataset uses "StatTrak™" — handled by
normalization. Souvenir items similarly use "Souvenir" prefix in both.

The API also provides per-entry `wears` list and `stattrak`/`souvenir` flags.
We expand each API entry into all its possible (prefix + base + wear) variants
to maximize join coverage.

Dependencies
------------
    requests  — HTTP GET for API fetch
    pandas    — DataFrame construction and manipulation
    json      — Local cache serialization
    os, time  — File age checking for cache invalidation
"""

import os
import json
import time
import re
import requests
import pandas as pd

# The working API endpoint (raw GitHub, not GitHub Pages which may 404)
API_URL = "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en/skins.json"
CACHE_PATH = "data/api_skins_cache.json"
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days


# ---------------------------------------------------------------------------
# Rarity ordinal mapping
# ---------------------------------------------------------------------------
# Maps the rarity name strings from the API to ordinal integers.
# Higher ordinal = rarer = typically more expensive.
# "Extraordinary" covers glove/knife rarities in the API.
RARITY_MAP = {
    'Consumer Grade':  1,
    'Industrial Grade': 2,
    'Mil-Spec Grade':  3,
    'Mil-Spec':        3,
    'Restricted':      4,
    'Classified':      5,
    'Covert':          6,
    'Contraband':      7,
    'Extraordinary':   6,  # Gloves/knives — treated as Covert-equivalent
}


def fetch_api_skins(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches the full skins catalog from the ByMykel CSGO-API.

    Implements a local file cache with a 7-day TTL. If the cache exists
    and is fresh, the API is not contacted.

    Parameters
    ----------
    force_refresh : bool
        If True, ignores the cache and fetches from the API regardless.

    Returns
    -------
    pd.DataFrame
        Raw API data with one row per skin entry. Columns include:
        'name', 'rarity', 'stattrak', 'souvenir', 'min_float',
        'max_float', 'collections', 'weapon', 'wears', etc.
    """
    # Check cache freshness
    if not force_refresh and os.path.exists(CACHE_PATH):
        cache_age = time.time() - os.path.getmtime(CACHE_PATH)
        if cache_age < CACHE_MAX_AGE_SECONDS:
            print(f"  Loading API data from cache ({CACHE_PATH}, "
                  f"age: {cache_age / 3600:.1f}h)")
            with open(CACHE_PATH, 'r') as f:
                raw_data = json.load(f)
            return pd.DataFrame(raw_data)

    # Fetch from API
    print(f"  Fetching skins data from API: {API_URL}")
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()
    raw_data = response.json()

    # Save to cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(raw_data, f)
    print(f"  Cached {len(raw_data)} skin entries to {CACHE_PATH}")

    return pd.DataFrame(raw_data)


def extract_api_features(raw_api_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts structured features from the raw API DataFrame.

    Transforms nested JSON fields (rarity dict, collections list) into
    flat, model-ready columns. Generates all possible item name variants
    (with/without StatTrak™/Souvenir prefix, with each wear condition)
    to maximize join coverage with the price dataset.

    Parameters
    ----------
    raw_api_df : pd.DataFrame
        Raw DataFrame from `fetch_api_skins()`.

    Returns
    -------
    pd.DataFrame
        Indexed by `item_name` (matching the price dataset format).
        Columns:
        - rarity_ordinal  : int (1=Consumer Grade … 7=Contraband), 0 if unknown
        - is_stattrak     : int (0 or 1)
        - is_souvenir     : int (0 or 1)
        - min_float       : float (minimum possible float value)
        - max_float       : float (maximum possible float value)
        - float_range     : float (max_float - min_float, measures wear variability)
        - collection      : str (primary collection name, or "None")
        - weapon_name     : str (weapon base name, e.g. "AK-47")
    """
    records = []

    for _, row in raw_api_df.iterrows():
        # Extract base skin name (without ★ prefix for knives/gloves)
        api_name = row.get('name', '')

        # Extract rarity
        rarity_dict = row.get('rarity', {})
        if isinstance(rarity_dict, dict):
            rarity_name = rarity_dict.get('name', '')
        else:
            rarity_name = ''
        rarity_ordinal = RARITY_MAP.get(rarity_name, 0)

        # StatTrak and Souvenir flags
        is_stattrak = int(bool(row.get('stattrak', False)))
        is_souvenir = int(bool(row.get('souvenir', False)))

        # Float caps
        min_float = float(row.get('min_float', 0.0) or 0.0)
        max_float = float(row.get('max_float', 1.0) or 1.0)
        float_range = max_float - min_float

        # Collection extraction
        collections = row.get('collections', [])
        if isinstance(collections, list) and len(collections) > 0:
            collection = collections[0].get('name', 'None') if isinstance(collections[0], dict) else 'None'
        else:
            collection = 'None'

        # Weapon name
        weapon_dict = row.get('weapon', {})
        if isinstance(weapon_dict, dict):
            weapon_name = weapon_dict.get('name', 'Unknown')
        else:
            weapon_name = 'Unknown'

        # Wears list
        wears = row.get('wears', [])
        wear_names = []
        if isinstance(wears, list):
            for w in wears:
                if isinstance(w, dict):
                    wear_names.append(w.get('name', ''))

        # If no wears listed, use a placeholder
        if not wear_names:
            wear_names = ['']

        # Strip the ★ prefix if present (the dataset doesn't always use it)
        base_name = api_name.lstrip('★').strip()

        # Generate all possible item_name variants to maximize join coverage
        # Dataset format examples:
        #   "AK-47 | Redline (Field-Tested)"
        #   "StatTrak™ AK-47 | Redline (Field-Tested)"
        #   "Souvenir P2000 | Amber Fade (Minimal Wear)"
        prefixes = ['']
        if is_stattrak:
            prefixes.append('StatTrak™ ')
        if is_souvenir:
            prefixes.append('Souvenir ')

        base_record = {
            'rarity_ordinal': rarity_ordinal,
            'is_stattrak': is_stattrak,
            'is_souvenir': is_souvenir,
            'min_float': min_float,
            'max_float': max_float,
            'float_range': float_range,
            'collection': collection,
            'weapon_name': weapon_name,
        }

        for prefix in prefixes:
            for wear in wear_names:
                if wear:
                    item_name = f"{prefix}{base_name} ({wear})"
                else:
                    item_name = f"{prefix}{base_name}"

                record = base_record.copy()
                record['item_name'] = item_name
                records.append(record)

    result_df = pd.DataFrame(records)

    # Deduplicate — some API entries may produce the same item_name
    result_df = result_df.drop_duplicates(subset='item_name', keep='first')
    result_df = result_df.set_index('item_name')

    return result_df
