"""
Unit Tests for Technical Indicators Module
"""

import pytest
import pandas as pd
import numpy as np
from src.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    add_all_indicators
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2023-01-01', periods=n, freq='H')

    data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.randn(n).cumsum(),
        'high': 102 + np.random.randn(n).cumsum(),
        'low': 98 + np.random.randn(n).cumsum(),
        'close': 100 + np.random.randn(n).cumsum(),
        'volume': np.random.randint(1000, 10000, n)
    })

    return data


def test_calculate_rsi(sample_data):
    """Test RSI calculation."""
    rsi = calculate_rsi(sample_data, period=14)

    # RSI should be between 0 and 100
    assert (rsi.dropna() >= 0).all()
    assert (rsi.dropna() <= 100).all()

    # RSI should have NaN values for the first few periods
    assert rsi.isna().sum() > 0


def test_calculate_macd(sample_data):
    """Test MACD calculation."""
    macd_df = calculate_macd(sample_data, fast_period=12, slow_period=26, signal_period=9)

    # Check that all columns exist
    assert 'macd' in macd_df.columns
    assert 'signal' in macd_df.columns
    assert 'histogram' in macd_df.columns

    # Check that histogram = macd - signal
    diff = macd_df['histogram'] - (macd_df['macd'] - macd_df['signal'])
    assert np.allclose(diff.dropna(), 0, atol=1e-10)


def test_calculate_bollinger_bands(sample_data):
    """Test Bollinger Bands calculation."""
    bb_df = calculate_bollinger_bands(sample_data, period=20, std_dev=2.0)

    # Check that all columns exist
    assert 'bb_upper' in bb_df.columns
    assert 'bb_middle' in bb_df.columns
    assert 'bb_lower' in bb_df.columns

    # Upper band should be greater than middle, middle greater than lower
    valid_data = bb_df.dropna()
    assert (valid_data['bb_upper'] >= valid_data['bb_middle']).all()
    assert (valid_data['bb_middle'] >= valid_data['bb_lower']).all()


def test_add_all_indicators(sample_data):
    """Test adding all indicators at once."""
    df_with_indicators = add_all_indicators(sample_data)

    # Check all expected columns are present
    expected_cols = ['rsi', 'macd', 'macd_signal', 'macd_histogram',
                     'bb_upper', 'bb_middle', 'bb_lower']

    for col in expected_cols:
        assert col in df_with_indicators.columns

    # Original data should not be modified
    assert len(df_with_indicators) == len(sample_data)
