"""
Technical Indicators Module

Implements RSI, MACD, and Bollinger Bands calculations.
"""

import pandas as pd
import numpy as np


def calculate_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        df: DataFrame with price data
        period: RSI period (default: 14)
        price_col: Column name for price data (default: 'close')

    Returns:
        Series with RSI values (0-100)
    """
    # Calculate price changes
    delta = df[price_col].diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate exponential moving averages
    avg_gains = gains.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        df: DataFrame with price data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        price_col: Column name for price data (default: 'close')

    Returns:
        DataFrame with columns: macd, signal, histogram
    """
    # Calculate EMAs
    ema_fast = df[price_col].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        df: DataFrame with price data
        period: Moving average period (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
        price_col: Column name for price data (default: 'close')

    Returns:
        DataFrame with columns: bb_upper, bb_middle, bb_lower
    """
    # Calculate middle band (SMA)
    middle_band = df[price_col].rolling(window=period).mean()

    # Calculate standard deviation
    std = df[price_col].rolling(window=period).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return pd.DataFrame({
        'bb_upper': upper_band,
        'bb_middle': middle_band,
        'bb_lower': lower_band
    })


def add_all_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Add all technical indicators to the dataframe.

    Args:
        df: DataFrame with OHLCV data
        rsi_period: RSI period
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
        bb_period: Bollinger Bands period
        bb_std: Bollinger Bands standard deviation multiplier
        price_col: Column name for close price

    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()

    # Add RSI
    df['rsi'] = calculate_rsi(df, period=rsi_period, price_col=price_col)

    # Add MACD
    macd_df = calculate_macd(
        df,
        fast_period=macd_fast,
        slow_period=macd_slow,
        signal_period=macd_signal,
        price_col=price_col
    )
    df['macd'] = macd_df['macd']
    df['macd_signal'] = macd_df['signal']
    df['macd_histogram'] = macd_df['histogram']

    # Add Bollinger Bands
    bb_df = calculate_bollinger_bands(
        df,
        period=bb_period,
        std_dev=bb_std,
        price_col=price_col
    )
    df['bb_upper'] = bb_df['bb_upper']
    df['bb_middle'] = bb_df['bb_middle']
    df['bb_lower'] = bb_df['bb_lower']

    return df
