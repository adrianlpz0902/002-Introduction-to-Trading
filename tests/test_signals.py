"""
Unit Tests for Signal Generation Module
"""

import pytest
import pandas as pd
import numpy as np
from src.signals import (
    generate_rsi_signals,
    generate_macd_signals,
    generate_bb_signals,
    apply_confirmation_logic,
    create_signals
)


@pytest.fixture
def sample_data_with_indicators():
    """Create sample data with indicators."""
    n = 100
    np.random.seed(42)

    data = pd.DataFrame({
        'close': 100 + np.random.randn(n).cumsum(),
        'rsi': 30 + 40 * np.random.rand(n),  # RSI between 30-70
        'macd': np.random.randn(n),
        'macd_signal': np.random.randn(n),
        'bb_upper': 105 + np.random.randn(n),
        'bb_middle': 100 + np.random.randn(n),
        'bb_lower': 95 + np.random.randn(n)
    })

    return data


def test_generate_rsi_signals(sample_data_with_indicators):
    """Test RSI signal generation."""
    df = sample_data_with_indicators.copy()
    df['rsi'] = [25, 35, 45, 55, 65, 75] + [50] * (len(df) - 6)

    signals = generate_rsi_signals(df, oversold_threshold=30, overbought_threshold=70)

    # Check signal columns exist
    assert 'rsi_buy' in signals.columns
    assert 'rsi_sell' in signals.columns

    # First value should be buy signal (RSI = 25 < 30)
    assert signals['rsi_buy'].iloc[0] == True

    # Last oversold should trigger sell (RSI = 75 > 70)
    assert signals['rsi_sell'].iloc[5] == True


def test_generate_macd_signals(sample_data_with_indicators):
    """Test MACD signal generation."""
    df = sample_data_with_indicators.copy()

    # Create crossover scenario
    df['macd'] = [1, 2, 3, 2, 1, 0, -1, 0, 1]
    df['macd_signal'] = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    df = df.iloc[:9]

    signals = generate_macd_signals(df)

    # Check signal columns exist
    assert 'macd_buy' in signals.columns
    assert 'macd_sell' in signals.columns

    # Should detect crossover
    assert signals['macd_buy'].sum() > 0 or signals['macd_sell'].sum() > 0


def test_generate_bb_signals(sample_data_with_indicators):
    """Test Bollinger Bands signal generation."""
    df = sample_data_with_indicators.copy()

    # Set up scenario where price touches bands
    df['close'] = [94, 100, 106]  # lower, middle, upper
    df['bb_lower'] = [95, 95, 95]
    df['bb_upper'] = [105, 105, 105]
    df = df.iloc[:3]

    signals = generate_bb_signals(df)

    # Price at/below lower band should trigger buy
    assert signals['bb_buy'].iloc[0] == True

    # Price at/above upper band should trigger sell
    assert signals['bb_sell'].iloc[2] == True


def test_apply_confirmation_logic():
    """Test 2-of-3 confirmation logic."""
    df = pd.DataFrame({'close': [100, 101, 102]})

    rsi_signals = pd.DataFrame({
        'rsi_buy': [True, False, False],
        'rsi_sell': [False, False, True]
    })

    macd_signals = pd.DataFrame({
        'macd_buy': [True, True, False],
        'macd_sell': [False, False, True]
    })

    bb_signals = pd.DataFrame({
        'bb_buy': [False, True, False],
        'bb_sell': [False, False, True]
    })

    signals = apply_confirmation_logic(df, rsi_signals, macd_signals, bb_signals, min_confirmations=2)

    # First row: 2 buy signals (RSI + MACD) -> should trigger buy
    assert signals['buy_signal'].iloc[0] == True

    # Second row: 2 buy signals (MACD + BB) -> should trigger buy
    assert signals['buy_signal'].iloc[1] == True

    # Third row: 3 sell signals -> should trigger sell
    assert signals['sell_signal'].iloc[2] == True


def test_no_simultaneous_signals():
    """Test that simultaneous buy/sell signals are prevented."""
    df = pd.DataFrame({'close': [100]})

    # All indicators agree on both buy and sell (edge case)
    rsi_signals = pd.DataFrame({'rsi_buy': [True], 'rsi_sell': [True]})
    macd_signals = pd.DataFrame({'macd_buy': [True], 'macd_sell': [True]})
    bb_signals = pd.DataFrame({'bb_buy': [True], 'bb_sell': [True]})

    signals = apply_confirmation_logic(df, rsi_signals, macd_signals, bb_signals, min_confirmations=2)

    # Should not have both buy and sell signals
    assert not (signals['buy_signal'].iloc[0] and signals['sell_signal'].iloc[0])
