"""
Portfolio Management Module

Handles position tracking and portfolio state management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class Portfolio:
    """
    Manages portfolio state including cash, positions, and portfolio value.
    """

    def __init__(self, initial_cash: float = 10000.0, transaction_fee: float = 0.00125):
        """
        Initialize portfolio.

        Args:
            initial_cash: Starting cash amount (default: $10,000)
            transaction_fee: Transaction fee as decimal (default: 0.00125 = 0.125%)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.transaction_fee = transaction_fee

        # Position tracking
        self.position = 0.0  # Number of units held (positive = long, negative = short)
        self.entry_price = 0.0  # Price at which position was entered
        self.position_type = None  # 'long', 'short', or None

        # History tracking
        self.history = []

    def get_portfolio_value(self, current_price: float) -> float:
        """
        Calculate current portfolio value.

        Args:
            current_price: Current market price

        Returns:
            Total portfolio value (cash + position value)
        """
        if self.position == 0:
            return self.cash

        if self.position_type == 'long':
            # Long position: value = cash + (units * current_price)
            position_value = self.position * current_price
        elif self.position_type == 'short':
            # Short position: value = cash + (entry_value - current_value)
            # We borrowed units at entry_price and must return at current_price
            position_value = abs(self.position) * (self.entry_price - current_price)
        else:
            position_value = 0

        total_value = self.cash + position_value

        # Safety warning: check if portfolio is critically low
        if total_value < self.initial_cash * 0.20:
            print(f"⚠️  WARNING: Portfolio value (${total_value:.2f}) is below 20% of initial capital (${self.initial_cash:.2f})")

        return total_value

    def open_long(self, price: float, size: Optional[float] = None) -> bool:
        """
        Open a long position.

        Args:
            price: Entry price
            size: Position size in cash (if None, uses all available cash)

        Returns:
            True if position opened successfully, False otherwise
        """
        if self.position != 0:
            return False  # Already in a position

        if size is None:
            size = self.cash

        # Calculate transaction cost
        fee = size * self.transaction_fee
        available_after_fee = size - fee

        if available_after_fee <= 0:
            return False

        # Calculate number of units
        units = available_after_fee / price

        # Update portfolio
        self.position = units
        self.entry_price = price
        self.position_type = 'long'
        self.cash -= size

        return True

    def open_short(self, price: float, size: Optional[float] = None, max_size_pct: float = 0.50) -> bool:
        """
        Open a short position with conservative sizing.

        Args:
            price: Entry price
            size: Position size in cash (if None, uses percentage of available cash)
            max_size_pct: Maximum percentage of cash to use (default: 0.50 = 50%)

        Returns:
            True if position opened successfully, False otherwise
        """
        if self.position != 0:
            return False  # Already in a position

        if size is None:
            # Conservative sizing for shorts - use only max_size_pct of cash
            size = self.cash * max_size_pct
        else:
            # Limit size to max_size_pct of cash to prevent catastrophic losses
            size = min(size, self.cash * max_size_pct)

        # Calculate transaction cost
        fee = size * self.transaction_fee
        available_after_fee = size - fee

        if available_after_fee <= 0:
            return False

        # Calculate number of units (negative for short)
        # CRITICAL: Limit units to prevent portfolio from going negative
        max_safe_units = (self.cash * max_size_pct) / price
        units = min(available_after_fee / price, max_safe_units)

        # Update portfolio
        # Short: we receive cash from selling borrowed units
        self.position = -units
        self.entry_price = price
        self.position_type = 'short'
        self.cash += (units * price - fee)  # We receive the sale proceeds minus fee

        return True

    def close_position(self, price: float) -> bool:
        """
        Close current position.

        Args:
            price: Exit price

        Returns:
            True if position closed successfully, False otherwise
        """
        if self.position == 0:
            return False  # No position to close

        if self.position_type == 'long':
            # Sell units at current price
            proceeds = abs(self.position) * price
            fee = proceeds * self.transaction_fee
            self.cash += (proceeds - fee)

        elif self.position_type == 'short':
            # Buy back units at current price
            cost = abs(self.position) * price
            fee = cost * self.transaction_fee
            self.cash -= (cost + fee)

        # Reset position
        self.position = 0.0
        self.entry_price = 0.0
        self.position_type = None

        return True

    def record_state(self, timestamp, price: float, signal: Optional[str] = None):
        """
        Record current portfolio state to history.

        Args:
            timestamp: Current timestamp
            price: Current market price
            signal: Trading signal if any ('buy', 'sell', or None)
        """
        self.history.append({
            'timestamp': timestamp,
            'price': price,
            'cash': self.cash,
            'position': self.position,
            'position_type': self.position_type,
            'portfolio_value': self.get_portfolio_value(price),
            'signal': signal
        })

    def get_history_df(self) -> pd.DataFrame:
        """
        Get portfolio history as DataFrame.

        Returns:
            DataFrame with portfolio history
        """
        return pd.DataFrame(self.history)

    def get_statistics(self) -> Dict:
        """
        Get portfolio statistics.

        Returns:
            Dictionary with portfolio statistics
        """
        if not self.history:
            return {}

        df = self.get_history_df()
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash

        return {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_records': len(self.history)
        }
