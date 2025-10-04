"""
Data Loading and Preprocessing Module

Handles loading OHLCV data from CSV and splitting into train/test/validation sets.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_and_preprocess_data(
    filepath: str,
    train_ratio: float = 0.60,
    test_ratio: float = 0.20,
    val_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CSV data and split into train/test/validation sets.

    Args:
        filepath: Path to the CSV file containing OHLCV data
        train_ratio: Proportion of data for training (default: 0.60)
        test_ratio: Proportion of data for testing (default: 0.20)
        val_ratio: Proportion of data for validation (default: 0.20)

    Returns:
        Tuple of (train_data, test_data, val_data) as pandas DataFrames

    Raises:
        ValueError: If ratios don't sum to 1.0
        FileNotFoundError: If CSV file doesn't exist
    """
    # Validate ratios
    if not np.isclose(train_ratio + test_ratio + val_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + test_ratio + val_ratio}")

    # Load data (skip first row if it's a header/URL, handle mixed types)
    df = pd.read_csv(filepath, skiprows=1, low_memory=False)

    # Clean data
    df = clean_data(df)

    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    test_end = int(n * (train_ratio + test_ratio))

    # Split data
    train_data = df.iloc[:train_end].copy()
    test_data = df.iloc[train_end:test_end].copy()
    val_data = df.iloc[test_end:].copy()

    print(f"Data loaded: {len(df)} total rows")
    print(f"  Train: {len(train_data)} rows ({train_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_data)} rows ({test_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_data)} rows ({val_ratio*100:.0f}%)")

    return train_data, test_data, val_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate OHLCV data.

    Args:
        df: Raw dataframe with OHLCV data

    Returns:
        Cleaned dataframe
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure timestamp column exists (common names)
    timestamp_cols = ['timestamp', 'date', 'datetime', 'time', 'Timestamp', 'Date']
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            # Try parsing as datetime string first, then as unix timestamp
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

            # If that didn't work and values look like unix timestamps, try unit='ms'
            if df[timestamp_col].isna().all():
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms', errors='coerce')

        # Sort by timestamp (descending to get most recent first, then reverse)
        df = df.sort_values(timestamp_col, ascending=False).reset_index(drop=True)

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    # For OHLCV data, forward fill is appropriate for small gaps
    df = df.ffill()

    # Drop any remaining NaN rows
    df = df.dropna()

    # Validate OHLCV columns exist (case-insensitive)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df_cols_lower = {col.lower(): col for col in df.columns}

    # Handle volume column variations (Volume BTC, Volume USDT, etc.)
    if 'volume' not in df_cols_lower:
        # Try common variations
        for col in df.columns:
            if 'volume' in col.lower():
                df_cols_lower['volume'] = col
                break

    for req_col in required_cols:
        if req_col not in df_cols_lower:
            raise ValueError(f"Required column '{req_col}' not found in data. Available columns: {list(df.columns)}")

    # Validate price logic: high >= low, close/open between high and low
    df_std = df.copy()
    for col in required_cols:
        actual_col = df_cols_lower[col]
        df_std[col] = df[actual_col]

    invalid_rows = (
        (df_std['high'] < df_std['low']) |
        (df_std['close'] > df_std['high']) |
        (df_std['close'] < df_std['low']) |
        (df_std['open'] > df_std['high']) |
        (df_std['open'] < df_std['low'])
    )

    if invalid_rows.sum() > 0:
        print(f"Warning: Removing {invalid_rows.sum()} rows with invalid OHLC values")
        df = df[~invalid_rows].reset_index(drop=True)

    return df


def get_ohlcv_columns(df: pd.DataFrame) -> dict:
    """
    Get the actual column names for OHLCV data (handles case variations).

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary mapping standard names to actual column names
    """
    df_cols_lower = {col.lower(): col for col in df.columns}

    return {
        'open': df_cols_lower.get('open'),
        'high': df_cols_lower.get('high'),
        'low': df_cols_lower.get('low'),
        'close': df_cols_lower.get('close'),
        'volume': df_cols_lower.get('volume')
    }
