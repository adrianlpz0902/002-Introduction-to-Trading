"""
Backtesting Engine Module

Simulates trading strategy with realistic constraints and transaction costs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .portfolio import Portfolio


def run_backtest(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    transaction_fee: float = 0.00125,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    position_size: Optional[float] = None,
    price_col: str = 'close'
) -> Tuple[Portfolio, pd.DataFrame]:
    """
    Run backtesting simulation on data with signals.

    Args:
        df: DataFrame with signals ('buy_signal', 'sell_signal') and price data
        initial_cash: Starting cash (default: $10,000)
        transaction_fee: Fee as decimal (default: 0.00125 = 0.125%)
        stop_loss: Stop loss percentage (e.g., 0.02 = 2%)
        take_profit: Take profit percentage (e.g., 0.03 = 3%)
        position_size: Fixed position size (if None, uses all cash)
        price_col: Column name for price data

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
    timestamp_cols = ['timestamp', 'date', 'datetime', 'time']
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break

    # Simulate trading
    for idx, row in df.iterrows():
        price = row[price_col]
        timestamp = row[timestamp_col] if timestamp_col else idx

        # Check stop loss and take profit
        if portfolio.position != 0:
            if _check_exit_conditions(portfolio, price, stop_loss, take_profit):
                portfolio.close_position(price)
                portfolio.record_state(timestamp, price, signal='exit_sl_tp')
                continue

        # Process signals
        if row['buy_signal'] and portfolio.position == 0:
            # Open long position
            if portfolio.open_long(price, size=position_size):
                portfolio.record_state(timestamp, price, signal='buy')
            else:
                portfolio.record_state(timestamp, price, signal=None)

        elif row['sell_signal'] and portfolio.position == 0:
            # Open short position
            if portfolio.open_short(price, size=position_size):
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
