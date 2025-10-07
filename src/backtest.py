"""
Backtesting Engine Module

Simulates trading strategy with realistic constraints and transaction costs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .portfolio import Portfolio


def calculate_position_size(
    portfolio: Portfolio,
    position_type: str,
    long_pct: float = 0.95,
    short_pct: float = 0.50
) -> float:
    """
    Calculate dynamic position size based on available cash and position type.

    Args:
        portfolio: Portfolio object
        position_type: 'long' or 'short'
        long_pct: Percentage of cash to use for long positions (default: 0.95)
        short_pct: Percentage of cash to use for short positions (default: 0.50)

    Returns:
        Position size in cash
    """
    if position_type == 'long':
        return portfolio.cash * long_pct
    elif position_type == 'short':
        return portfolio.cash * short_pct
    else:
        return 0.0


def run_backtest(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    transaction_fee: float = 0.00125,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    position_size: Optional[float] = None,
    price_col: str = 'close',
    long_position_size_pct: float = 0.95,
    short_position_size_pct: float = 0.50,
    max_drawdown_threshold: float = 0.30,
    short_stop_loss_multiplier: float = 0.75
) -> Tuple[Portfolio, pd.DataFrame]:
    """
    Run backtesting simulation on data with signals.

    Args:
        df: DataFrame with signals ('buy_signal', 'sell_signal') and price data
        initial_cash: Starting cash (default: $10,000)
        transaction_fee: Fee as decimal (default: 0.00125 = 0.125%)
        stop_loss: Stop loss percentage (e.g., 0.02 = 2%)
        take_profit: Take profit percentage (e.g., 0.03 = 3%)
        position_size: Fixed position size (DEPRECATED - use dynamic sizing instead)
        price_col: Column name for price data
        long_position_size_pct: Percentage of cash for long positions (default: 0.95)
        short_position_size_pct: Percentage of cash for short positions (default: 0.50)
        max_drawdown_threshold: Circuit breaker threshold (default: 0.30 = 30%)
        short_stop_loss_multiplier: Tighter stop-loss for shorts (default: 0.75)

    Returns:
        Tuple of (Portfolio object, DataFrame with returns)
    """
    # Initialize portfolio
    portfolio = Portfolio(initial_cash=initial_cash, transaction_fee=transaction_fee)

    # Ensure we have required columns
    required_cols = ['buy_signal', 'sell_signal', price_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")

    # Get timestamp column if exists
    timestamp_cols = ['timestamp', 'date', 'datetime', 'time', 'Date']
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break

    # Track peak portfolio value for circuit breaker
    peak_value = initial_cash
    circuit_breaker_triggered = False

    # Simulate trading
    for idx, row in df.iterrows():
        price = row[price_col]
        timestamp = row[timestamp_col] if timestamp_col else idx

        current_value = portfolio.get_portfolio_value(price)

        # Update peak value
        if current_value > peak_value:
            peak_value = current_value

        # CIRCUIT BREAKER: Stop trading if drawdown exceeds threshold
        current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
        if current_drawdown >= max_drawdown_threshold and not circuit_breaker_triggered:
            circuit_breaker_triggered = True
            print(f"🛑 CIRCUIT BREAKER TRIGGERED: Drawdown {current_drawdown*100:.1f}% >= {max_drawdown_threshold*100:.1f}% threshold")
            # Close any open position
            if portfolio.position != 0:
                portfolio.close_position(price)
            portfolio.record_state(timestamp, price, signal='circuit_breaker')
            continue

        # If circuit breaker is active, only record state (no trading)
        if circuit_breaker_triggered:
            portfolio.record_state(timestamp, price, signal=None)
            continue

        # Safety check: prevent negative portfolio value
        if current_value <= 0:
            print(f"🚨 CRITICAL: Portfolio value is ${current_value:.2f} - halting trading")
            portfolio.record_state(timestamp, price, signal='critical_halt')
            break

        # Determine stop-loss for current position
        active_stop_loss = stop_loss
        if portfolio.position_type == 'short' and stop_loss is not None:
            # Tighter stop-loss for shorts
            active_stop_loss = stop_loss * short_stop_loss_multiplier

        # Check stop loss and take profit
        if portfolio.position != 0:
            if _check_exit_conditions(portfolio, price, active_stop_loss, take_profit):
                portfolio.close_position(price)
                portfolio.record_state(timestamp, price, signal='exit_sl_tp')
                continue

        # Process signals with dynamic position sizing
        if row['buy_signal'] and portfolio.position == 0:
            # Open long position with dynamic sizing
            size = calculate_position_size(portfolio, 'long', long_position_size_pct, short_position_size_pct)
            if portfolio.open_long(price, size=size):
                portfolio.record_state(timestamp, price, signal='buy')
            else:
                portfolio.record_state(timestamp, price, signal=None)

        elif row['sell_signal'] and portfolio.position == 0:
            # Open short position with dynamic sizing
            size = calculate_position_size(portfolio, 'short', long_position_size_pct, short_position_size_pct)
            if portfolio.open_short(price, size=size, max_size_pct=short_position_size_pct):
                portfolio.record_state(timestamp, price, signal='sell')
            else:
                portfolio.record_state(timestamp, price, signal=None)

        elif row['sell_signal'] and portfolio.position_type == 'long':
            # Close long position
            portfolio.close_position(price)
            portfolio.record_state(timestamp, price, signal='close_long')

        elif row['buy_signal'] and portfolio.position_type == 'short':
            # Close short position
            portfolio.close_position(price)
            portfolio.record_state(timestamp, price, signal='close_short')

        else:
            # No action, just record state
            portfolio.record_state(timestamp, price, signal=None)

    # Close any remaining position at the end
    if portfolio.position != 0:
        final_price = df[price_col].iloc[-1]
        final_timestamp = df[timestamp_col].iloc[-1] if timestamp_col else len(df) - 1
        portfolio.close_position(final_price)
        portfolio.record_state(final_timestamp, final_price, signal='close_final')

    # Calculate returns
    history_df = portfolio.get_history_df()
    history_df['returns'] = history_df['portfolio_value'].pct_change()

    return portfolio, history_df


def _check_exit_conditions(
    portfolio: Portfolio,
    current_price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float]
) -> bool:
    """
    Check if stop loss or take profit conditions are met.

    Args:
        portfolio: Portfolio object
        current_price: Current market price
        stop_loss: Stop loss threshold (as decimal)
        take_profit: Take profit threshold (as decimal)

    Returns:
        True if exit conditions met, False otherwise
    """
    if portfolio.position == 0:
        return False

    entry_price = portfolio.entry_price

    if portfolio.position_type == 'long':
        # Long position
        pnl_pct = (current_price - entry_price) / entry_price

        # Check stop loss
        if stop_loss is not None and pnl_pct <= -stop_loss:
            return True

        # Check take profit
        if take_profit is not None and pnl_pct >= take_profit:
            return True

    elif portfolio.position_type == 'short':
        # Short position (gains when price falls)
        pnl_pct = (entry_price - current_price) / entry_price

        # Check stop loss
        if stop_loss is not None and pnl_pct <= -stop_loss:
            return True

        # Check take profit
        if take_profit is not None and pnl_pct >= take_profit:
            return True

    return False


def calculate_trade_statistics(history_df: pd.DataFrame) -> Dict:
    """
    Calculate statistics about trades executed.

    Args:
        history_df: Portfolio history DataFrame

    Returns:
        Dictionary with trade statistics
    """
    # Find all trade entries and exits
    trades = history_df[history_df['signal'].notna() & (history_df['signal'] != '')]

    # Count buy and sell signals
    buy_signals = trades[trades['signal'].str.contains('buy', case=False, na=False)]
    sell_signals = trades[trades['signal'].str.contains('sell', case=False, na=False)]
    close_signals = trades[trades['signal'].str.contains('close|exit', case=False, na=False)]

    total_trades = len(buy_signals) + len(sell_signals)

    return {
        'total_signals': len(trades),
        'total_trades': total_trades,
        'long_entries': len(buy_signals),
        'short_entries': len(sell_signals),
        'exits': len(close_signals)
    }
