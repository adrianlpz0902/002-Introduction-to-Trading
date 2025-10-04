"""
Signal Generation Module

Implements 2-of-3 confirmation logic for trading signals.
"""

import pandas as pd
import numpy as np


def generate_rsi_signals(
    df: pd.DataFrame,
    oversold_threshold: float = 30,
    overbought_threshold: float = 70
) -> pd.DataFrame:
    """
    Generate buy/sell signals from RSI.

    Args:
        df: DataFrame with 'rsi' column
        oversold_threshold: RSI level for buy signal (default: 30)
        overbought_threshold: RSI level for sell signal (default: 70)

    Returns:
        DataFrame with rsi_buy and rsi_sell boolean columns
    """
    signals = pd.DataFrame(index=df.index)
    signals['rsi_buy'] = df['rsi'] < oversold_threshold
    signals['rsi_sell'] = df['rsi'] > overbought_threshold

    return signals


def generate_macd_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy/sell signals from MACD crossovers.

    Args:
        df: DataFrame with 'macd' and 'macd_signal' columns

    Returns:
        DataFrame with macd_buy and macd_sell boolean columns
    """
    signals = pd.DataFrame(index=df.index)

    # MACD crosses above signal line = buy
    # MACD crosses below signal line = sell
    macd_diff = df['macd'] - df['macd_signal']
    macd_diff_prev = macd_diff.shift(1)

    signals['macd_buy'] = (macd_diff > 0) & (macd_diff_prev <= 0)
    signals['macd_sell'] = (macd_diff < 0) & (macd_diff_prev >= 0)

    return signals


def generate_bb_signals(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Generate buy/sell signals from Bollinger Bands.

    Args:
        df: DataFrame with Bollinger Bands columns and price
        price_col: Column name for price (default: 'close')

    Returns:
        DataFrame with bb_buy and bb_sell boolean columns
    """
    signals = pd.DataFrame(index=df.index)

    # Price touches/crosses lower band = buy
    # Price touches/crosses upper band = sell
    signals['bb_buy'] = df[price_col] <= df['bb_lower']
    signals['bb_sell'] = df[price_col] >= df['bb_upper']

    return signals


def apply_confirmation_logic(
    df: pd.DataFrame,
    rsi_signals: pd.DataFrame,
    macd_signals: pd.DataFrame,
    bb_signals: pd.DataFrame,
    min_confirmations: int = 2
) -> pd.DataFrame:
    """
    Apply 2-of-3 confirmation logic to generate final signals.

    Args:
        df: Original DataFrame
        rsi_signals: RSI signals DataFrame
        macd_signals: MACD signals DataFrame
        bb_signals: Bollinger Bands signals DataFrame
        min_confirmations: Minimum number of indicators that must agree (default: 2)

    Returns:
        DataFrame with 'buy_signal' and 'sell_signal' boolean columns
    """
    signals = pd.DataFrame(index=df.index)

    # Count buy signals
    buy_count = (
        rsi_signals['rsi_buy'].astype(int) +
        macd_signals['macd_buy'].astype(int) +
        bb_signals['bb_buy'].astype(int)
    )

    # Count sell signals
    sell_count = (
        rsi_signals['rsi_sell'].astype(int) +
        macd_signals['macd_sell'].astype(int) +
        bb_signals['bb_sell'].astype(int)
    )

    # Generate final signals (2-of-3 rule)
    signals['buy_signal'] = buy_count >= min_confirmations
    signals['sell_signal'] = sell_count >= min_confirmations

    # Prevent simultaneous buy and sell signals
    simultaneous = signals['buy_signal'] & signals['sell_signal']
    if simultaneous.any():
        # In case of conflict, no signal
        signals.loc[simultaneous, 'buy_signal'] = False
        signals.loc[simultaneous, 'sell_signal'] = False

    return signals


def create_signals(
    df: pd.DataFrame,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    min_confirmations: int = 2,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Main function to create all signals with 2-of-3 confirmation.

    Args:
        df: DataFrame with all indicators already calculated
        rsi_oversold: RSI oversold threshold
        rsi_overbought: RSI overbought threshold
        min_confirmations: Minimum indicators that must agree
        price_col: Column name for close price

    Returns:
        DataFrame with all signals added
    """
    df = df.copy()

    # Generate individual indicator signals
    rsi_signals = generate_rsi_signals(df, rsi_oversold, rsi_overbought)
    macd_signals = generate_macd_signals(df)
    bb_signals = generate_bb_signals(df, price_col)

    # Add individual signals to dataframe (for debugging/analysis)
    df['rsi_buy'] = rsi_signals['rsi_buy']
    df['rsi_sell'] = rsi_signals['rsi_sell']
    df['macd_buy'] = macd_signals['macd_buy']
    df['macd_sell'] = macd_signals['macd_sell']
    df['bb_buy'] = bb_signals['bb_buy']
    df['bb_sell'] = bb_signals['bb_sell']

    # Apply confirmation logic
    final_signals = apply_confirmation_logic(
        df, rsi_signals, macd_signals, bb_signals, min_confirmations
    )

    df['buy_signal'] = final_signals['buy_signal']
    df['sell_signal'] = final_signals['sell_signal']

    return df
