"""
Simple Backtesting Engine with Risk Management

Clean implementation following class structure from AAA copy.ipynb
Supports both LONG and SHORT positions with stop loss / take profit
Includes risk management to prevent >100% drawdown
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Position:
    """Position tracking for long/short"""
    position_type: str  # 'long' or 'short'
    price: float
    shares: float
    stop_loss: float
    take_profit: float
    timestamp: str


def run_backtest(
    df: pd.DataFrame,
    initial_cash: float = 10000.0,
    transaction_fee: float = 0.00125,
    stop_loss: float = 0.02,
    take_profit: float = 0.03,
    position_size_pct: float = 0.95,
    price_col: str = 'Close',
    max_positions: int = 1,
    max_position_size_pct: float = 0.30,
    min_portfolio_pct: float = 0.20,
    max_drawdown_limit: float = 0.40
) -> Tuple[pd.DataFrame, dict]:
    """
    Simple backtest with risk management - supports LONG and SHORT positions.

    Args:
        df: DataFrame with 'buy_signal', 'sell_signal' columns
        initial_cash: Starting capital
        transaction_fee: Fee as decimal (0.00125 = 0.125%)
        stop_loss: Stop loss percentage (0.02 = 2%)
        take_profit: Take profit percentage (0.03 = 3%)
        position_size_pct: Percentage of cash to use per trade (0.95 = 95%)
        price_col: Name of price column
        max_positions: Maximum concurrent positions (default: 1)
        max_position_size_pct: Max position size as % of initial capital (default: 0.30)
        min_portfolio_pct: Circuit breaker threshold (default: 0.20 = 20%)
        max_drawdown_limit: Max allowed drawdown before circuit breaker (default: 0.40 = 40%)

    Returns:
        (history_df, final_portfolio)
    """
    # Initialize
    cash = initial_cash
    active_long_positions: List[Position] = []
    active_short_positions: List[Position] = []

    # Risk management tracking
    peak_portfolio_value = initial_cash
    circuit_breaker_triggered = False

    # History tracking
    history = []

    for idx, row in df.iterrows():
        price = row[price_col]
        timestamp = str(idx)

        # --- Check LONG positions for exit conditions ---
        for position in active_long_positions.copy():
            # Long: profit if price goes UP, loss if price goes DOWN
            if price >= position.take_profit or price <= position.stop_loss:
                # Close long position
                cash += price * position.shares * (1 - transaction_fee)
                active_long_positions.remove(position)

        # --- Check SHORT positions for exit conditions ---
        for position in active_short_positions.copy():
            # Short: profit if price goes DOWN, loss if price goes UP
            if price <= position.take_profit or price >= position.stop_loss:
                # Close short position
                # Return reserved margin + profit/loss
                profit_or_loss = (position.price - price) * position.shares
                reserved_margin = position.price * position.shares * (1 + transaction_fee)
                cash += reserved_margin + profit_or_loss - (price * position.shares * transaction_fee)
                active_short_positions.remove(position)

        # Calculate current portfolio value
        long_value = sum(pos.shares * price for pos in active_long_positions)
        short_value = sum((pos.price - price) * pos.shares for pos in active_short_positions)
        portfolio_value = cash + long_value + short_value

        # --- Risk Management: Circuit Breakers ---
        # Update peak portfolio value
        if portfolio_value > peak_portfolio_value:
            peak_portfolio_value = portfolio_value

        # Calculate current drawdown
        current_drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value if peak_portfolio_value > 0 else 0

        # Check circuit breakers
        if not circuit_breaker_triggered:
            # Circuit breaker 1: Portfolio floor
            if portfolio_value < initial_cash * min_portfolio_pct:
                circuit_breaker_triggered = True
                # Close all positions immediately
                for pos in active_long_positions.copy():
                    cash += price * pos.shares * (1 - transaction_fee)
                    active_long_positions.remove(pos)
                for pos in active_short_positions.copy():
                    profit_or_loss = (pos.price - price) * pos.shares
                    reserved_margin = pos.price * pos.shares * (1 + transaction_fee)
                    cash += reserved_margin + profit_or_loss - (price * pos.shares * transaction_fee)
                    active_short_positions.remove(pos)

            # Circuit breaker 2: Max drawdown
            elif current_drawdown > max_drawdown_limit:
                circuit_breaker_triggered = True
                # Close all positions immediately
                for pos in active_long_positions.copy():
                    cash += price * pos.shares * (1 - transaction_fee)
                    active_long_positions.remove(pos)
                for pos in active_short_positions.copy():
                    profit_or_loss = (pos.price - price) * pos.shares
                    reserved_margin = pos.price * pos.shares * (1 + transaction_fee)
                    cash += reserved_margin + profit_or_loss - (price * pos.shares * transaction_fee)
                    active_short_positions.remove(pos)

        # Skip trading if circuit breaker triggered
        if circuit_breaker_triggered:
            # Recalculate portfolio value after forced closures
            long_value = sum(pos.shares * price for pos in active_long_positions)
            short_value = sum((pos.price - price) * pos.shares for pos in active_short_positions)
            portfolio_value = cash + long_value + short_value

            history.append({
                'timestamp': timestamp,
                'price': price,
                'cash': cash,
                'long_value': long_value,
                'short_value': short_value,
                'portfolio_value': portfolio_value,
                'num_long': len(active_long_positions),
                'num_short': len(active_short_positions),
                'signal': 'circuit_breaker',
                'position_type': 'neutral'
            })
            continue

        # --- Check for BUY signal (open LONG) ---
        if row['buy_signal']:
            # RULE 1: Only one position at a time
            total_positions = len(active_long_positions) + len(active_short_positions)

            if total_positions >= max_positions:
                # Close existing positions first
                for pos in active_long_positions.copy():
                    cash += price * pos.shares * (1 - transaction_fee)
                    active_long_positions.remove(pos)
                for pos in active_short_positions.copy():
                    profit_or_loss = (pos.price - price) * pos.shares
                    reserved_margin = pos.price * pos.shares * (1 + transaction_fee)
                    cash += reserved_margin + profit_or_loss - (price * pos.shares * transaction_fee)
                    active_short_positions.remove(pos)

            # RULE 2: Position size limited by initial capital
            max_position_value = initial_cash * max_position_size_pct
            available_cash = min(cash * position_size_pct, max_position_value)

            cost_per_share = price * (1 + transaction_fee)
            shares = available_cash / cost_per_share
            total_cost = cost_per_share * shares

            if cash >= total_cost and total_cost > 0:
                cash -= total_cost

                pos = Position(
                    position_type='long',
                    price=price,
                    shares=shares,
                    stop_loss=price * (1 - stop_loss),
                    take_profit=price * (1 + take_profit),
                    timestamp=timestamp
                )
                active_long_positions.append(pos)

        # --- Check for SELL signal (open SHORT) ---
        if row['sell_signal']:
            # RULE 1: Only one position at a time
            total_positions = len(active_long_positions) + len(active_short_positions)

            if total_positions >= max_positions:
                # Close existing positions first
                for pos in active_long_positions.copy():
                    cash += price * pos.shares * (1 - transaction_fee)
                    active_long_positions.remove(pos)
                for pos in active_short_positions.copy():
                    profit_or_loss = (pos.price - price) * pos.shares
                    reserved_margin = pos.price * pos.shares * (1 + transaction_fee)
                    cash += reserved_margin + profit_or_loss - (price * pos.shares * transaction_fee)
                    active_short_positions.remove(pos)

            # RULE 2: Position size limited by initial capital
            max_position_value = initial_cash * max_position_size_pct
            available_cash = min(cash * position_size_pct, max_position_value)

            cost_per_share = price * (1 + transaction_fee)
            shares = available_cash / cost_per_share
            total_cost = cost_per_share * shares

            if cash >= total_cost and total_cost > 0:
                # Reserve cash for shorting (margin requirement)
                cash -= total_cost

                pos = Position(
                    position_type='short',
                    price=price,
                    shares=shares,
                    stop_loss=price * (1 + stop_loss),  # Reverse for short
                    take_profit=price * (1 - take_profit),  # Reverse for short
                    timestamp=timestamp
                )
                active_short_positions.append(pos)

        # Recalculate portfolio value after any trades
        long_value = sum(pos.shares * price for pos in active_long_positions)
        short_value = sum((pos.price - price) * pos.shares for pos in active_short_positions)
        portfolio_value = cash + long_value + short_value

        # Record history
        history.append({
            'timestamp': timestamp,
            'price': price,
            'cash': cash,
            'long_value': long_value,
            'short_value': short_value,
            'portfolio_value': portfolio_value,
            'num_long': len(active_long_positions),
            'num_short': len(active_short_positions),
            'signal': 'buy' if row['buy_signal'] else ('sell' if row['sell_signal'] else 'hold'),
            'position_type': 'long' if len(active_long_positions) > 0 else ('short' if len(active_short_positions) > 0 else 'neutral')
        })

    # Close all positions at end
    final_price = df[price_col].iloc[-1]

    for pos in active_long_positions:
        cash += final_price * pos.shares * (1 - transaction_fee)

    for pos in active_short_positions:
        profit = (pos.price - final_price) * pos.shares
        reserved_margin = pos.price * pos.shares * (1 + transaction_fee)
        cash += reserved_margin + profit - (final_price * pos.shares * transaction_fee)

    active_long_positions = []
    active_short_positions = []

    # Create history DataFrame
    history_df = pd.DataFrame(history)

    # Final portfolio state
    final_portfolio = {
        'initial_cash': initial_cash,
        'final_cash': cash,
        'cash': cash,
        'transaction_fee': transaction_fee,
        'circuit_breaker_triggered': circuit_breaker_triggered
    }

    return history_df, final_portfolio
