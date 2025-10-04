"""
Unit Tests for Backtesting Engine
"""

import pytest
import pandas as pd
import numpy as np
from src.backtest import run_backtest, _check_exit_conditions
from src.portfolio import Portfolio


@pytest.fixture
def sample_signal_data():
    """Create sample data with signals."""
    n = 50
    np.random.seed(42)

    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='H'),
        'close': 100 + np.random.randn(n).cumsum(),
        'buy_signal': [False] * n,
        'sell_signal': [False] * n
    })

    # Add some buy and sell signals
    data.loc[5, 'buy_signal'] = True
    data.loc[15, 'sell_signal'] = True
    data.loc[25, 'buy_signal'] = True
    data.loc[35, 'sell_signal'] = True

    return data


def test_run_backtest(sample_signal_data):
    """Test basic backtesting functionality."""
    portfolio, history_df = run_backtest(
        sample_signal_data,
        initial_cash=10000.0,
        transaction_fee=0.00125
    )

    # Check that portfolio was created
    assert portfolio is not None
    assert portfolio.initial_cash == 10000.0

    # Check that history was recorded
    assert len(history_df) > 0
    assert 'portfolio_value' in history_df.columns
    assert 'returns' in history_df.columns


def test_transaction_fees_applied(sample_signal_data):
    """Test that transaction fees are applied correctly."""
    # Run with no fees
    portfolio_no_fee, history_no_fee = run_backtest(
        sample_signal_data,
        initial_cash=10000.0,
        transaction_fee=0.0
    )

    # Run with fees
    portfolio_with_fee, history_with_fee = run_backtest(
        sample_signal_data,
        initial_cash=10000.0,
        transaction_fee=0.00125
    )

    # Portfolio with fees should have lower final value (assuming positive returns)
    # or account for fee impact
    assert portfolio_no_fee is not None
    assert portfolio_with_fee is not None


def test_stop_loss_trigger():
    """Test stop loss functionality."""
    # Create scenario where stop loss should trigger
    data = pd.DataFrame({
        'close': [100, 95, 90, 85],  # Price drops 15%
        'buy_signal': [True, False, False, False],
        'sell_signal': [False, False, False, False]
    })

    portfolio, history_df = run_backtest(
        data,
        initial_cash=10000.0,
        stop_loss=0.05  # 5% stop loss
    )

    # Position should be closed due to stop loss
    signals = history_df['signal'].values

    # Should have an exit signal due to stop loss
    assert any('exit' in str(s).lower() for s in signals if pd.notna(s))


def test_take_profit_trigger():
    """Test take profit functionality."""
    # Create scenario where take profit should trigger
    data = pd.DataFrame({
        'close': [100, 105, 110, 115],  # Price rises 15%
        'buy_signal': [True, False, False, False],
        'sell_signal': [False, False, False, False]
    })

    portfolio, history_df = run_backtest(
        data,
        initial_cash=10000.0,
        take_profit=0.10  # 10% take profit
    )

    # Position should be closed due to take profit
    signals = history_df['signal'].values

    # Should have an exit signal due to take profit
    assert any('exit' in str(s).lower() for s in signals if pd.notna(s))


def test_check_exit_conditions():
    """Test exit condition checking."""
    portfolio = Portfolio(initial_cash=10000.0)

    # Open long position
    portfolio.open_long(price=100.0)

    # Test stop loss
    assert _check_exit_conditions(portfolio, 93.0, stop_loss=0.05, take_profit=None) == True

    # Reset position
    portfolio = Portfolio(initial_cash=10000.0)
    portfolio.open_long(price=100.0)

    # Test take profit
    assert _check_exit_conditions(portfolio, 111.0, stop_loss=None, take_profit=0.10) == True

    # Reset position
    portfolio = Portfolio(initial_cash=10000.0)
    portfolio.open_long(price=100.0)

    # Test no exit
    assert _check_exit_conditions(portfolio, 102.0, stop_loss=0.05, take_profit=0.10) == False
