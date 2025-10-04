"""
Unit Tests for Performance Metrics Module
"""

import pytest
import pandas as pd
import numpy as np
from src.metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_all_metrics
)


@pytest.fixture
def sample_portfolio_values():
    """Create sample portfolio values."""
    return pd.Series([10000, 10100, 10200, 10150, 10300, 10250, 10400])


@pytest.fixture
def sample_history():
    """Create sample portfolio history."""
    return pd.DataFrame({
        'price': [100, 105, 103, 108, 106],
        'signal': ['buy', None, 'close_long', 'sell', 'close_short'],
        'portfolio_value': [10000, 10500, 10300, 10800, 10600]
    })


def test_calculate_returns(sample_portfolio_values):
    """Test returns calculation."""
    returns = calculate_returns(sample_portfolio_values)

    # Returns should have one less value than portfolio values
    assert len(returns) == len(sample_portfolio_values) - 1

    # First return should be ~1%
    assert np.isclose(returns.iloc[0], 0.01, atol=0.001)


def test_calculate_sharpe_ratio(sample_portfolio_values):
    """Test Sharpe ratio calculation."""
    returns = calculate_returns(sample_portfolio_values)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)

    # Sharpe should be a finite number
    assert np.isfinite(sharpe)


def test_calculate_sortino_ratio(sample_portfolio_values):
    """Test Sortino ratio calculation."""
    returns = calculate_returns(sample_portfolio_values)
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)

    # Sortino should be a finite number or inf (if no downside)
    assert np.isfinite(sortino) or np.isinf(sortino)


def test_calculate_max_drawdown(sample_portfolio_values):
    """Test maximum drawdown calculation."""
    dd_info = calculate_max_drawdown(sample_portfolio_values)

    # Should have all required keys
    assert 'max_drawdown' in dd_info
    assert 'max_drawdown_pct' in dd_info
    assert 'peak_value' in dd_info
    assert 'trough_value' in dd_info

    # Max drawdown should be between 0 and 1
    assert 0 <= dd_info['max_drawdown'] <= 1

    # Peak should be >= trough
    assert dd_info['peak_value'] >= dd_info['trough_value']


def test_calculate_max_drawdown_with_decline():
    """Test max drawdown with a significant decline."""
    values = pd.Series([10000, 10500, 10200, 9500, 9000, 9500, 10000])
    dd_info = calculate_max_drawdown(values)

    # Max drawdown should be around 14.3% (from 10500 to 9000)
    expected_dd = (10500 - 9000) / 10500
    assert np.isclose(dd_info['max_drawdown'], expected_dd, atol=0.01)


def test_calculate_calmar_ratio(sample_portfolio_values):
    """Test Calmar ratio calculation."""
    returns = calculate_returns(sample_portfolio_values)
    calmar = calculate_calmar_ratio(returns, sample_portfolio_values)

    # Calmar should be a finite number or inf
    assert np.isfinite(calmar) or np.isinf(calmar)


def test_calculate_win_rate(sample_history):
    """Test win rate calculation."""
    wr_info = calculate_win_rate(sample_history)

    # Should have all required keys
    assert 'win_rate' in wr_info
    assert 'win_rate_pct' in wr_info
    assert 'total_trades' in wr_info
    assert 'winning_trades' in wr_info
    assert 'losing_trades' in wr_info

    # Win rate should be between 0 and 1
    assert 0 <= wr_info['win_rate'] <= 1

    # Total trades should equal winning + losing
    assert wr_info['total_trades'] == wr_info['winning_trades'] + wr_info['losing_trades']


def test_calculate_all_metrics(sample_portfolio_values, sample_history):
    """Test calculating all metrics at once."""
    metrics = calculate_all_metrics(
        sample_portfolio_values,
        sample_history,
        risk_free_rate=0.0
    )

    # Check all expected metrics are present
    expected_metrics = [
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
        'max_drawdown', 'max_drawdown_pct',
        'win_rate', 'win_rate_pct', 'total_trades',
        'total_return', 'total_return_pct'
    ]

    for metric in expected_metrics:
        assert metric in metrics


def test_edge_case_no_trades():
    """Test metrics with no trades."""
    history = pd.DataFrame({
        'price': [100, 101, 102],
        'signal': [None, None, None],
        'portfolio_value': [10000, 10000, 10000]
    })

    wr_info = calculate_win_rate(history)

    assert wr_info['total_trades'] == 0
    assert wr_info['win_rate'] == 0.0
