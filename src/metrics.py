"""
Performance Metrics Module

Calculates trading strategy performance metrics:
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Win Rate
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_returns(portfolio_values: pd.Series) -> pd.Series:
    """
    Calculate returns from portfolio values.

    Args:
        portfolio_values: Series of portfolio values over time

    Returns:
        Series of returns
    """
    return portfolio_values.pct_change().dropna()


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760  # Hourly data: 24 * 365
) -> float:
    """
    Calculate Sharpe Ratio.

    Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (default: 8760 for hourly)

    Returns:
        Sharpe Ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)

    # Calculate mean and std of excess returns
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0:
        return 0.0

    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760
) -> float:
    """
    Calculate Sortino Ratio.

    Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Deviation
    Similar to Sharpe but only penalizes downside volatility.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year

    Returns:
        Sortino Ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)

    # Calculate mean excess return
    mean_excess = excess_returns.mean()

    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    # Annualize
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)

    return sortino


def calculate_max_drawdown(portfolio_values: pd.Series) -> Dict[str, float]:
    """
    Calculate Maximum Drawdown.

    Max Drawdown = (Trough Value - Peak Value) / Peak Value

    Args:
        portfolio_values: Series of portfolio values over time

    Returns:
        Dictionary with max_drawdown, peak_value, trough_value
    """
    if len(portfolio_values) == 0:
        return {'max_drawdown': 0.0, 'peak_value': 0.0, 'trough_value': 0.0}

    # Calculate running maximum
    running_max = portfolio_values.expanding().max()

    # Calculate drawdown at each point
    drawdown = (portfolio_values - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdown.min()

    # Find peak and trough
    max_dd_idx = drawdown.idxmin()
    peak_idx = running_max[:max_dd_idx].idxmax() if max_dd_idx else 0

    return {
        'max_drawdown': abs(max_dd),
        'max_drawdown_pct': abs(max_dd) * 100,
        'peak_value': portfolio_values[peak_idx] if peak_idx else portfolio_values.iloc[0],
        'trough_value': portfolio_values[max_dd_idx] if max_dd_idx else portfolio_values.iloc[0]
    }


def calculate_calmar_ratio(
    returns: pd.Series,
    portfolio_values: pd.Series,
    periods_per_year: int = 8760
) -> float:
    """
    Calculate Calmar Ratio.

    Calmar Ratio = Annualized Return / Maximum Drawdown
    Higher is better. Measures return per unit of downside risk.

    Args:
        returns: Series of returns
        portfolio_values: Series of portfolio values
        periods_per_year: Number of periods in a year

    Returns:
        Calmar Ratio
    """
    if len(returns) < 2:
        return 0.0

    # Calculate annualized return
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # Calculate max drawdown
    max_dd_info = calculate_max_drawdown(portfolio_values)
    max_dd = max_dd_info['max_drawdown']

    if max_dd == 0:
        return np.inf if annualized_return > 0 else 0.0

    calmar = annualized_return / max_dd

    return calmar


def calculate_win_rate(history_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate win rate from trade history.

    Args:
        history_df: Portfolio history DataFrame with num_long/num_short columns

    Returns:
        Dictionary with win_rate, total_trades, winning_trades, losing_trades
    """
    # Count trades by tracking position changes
    trades = []
    prev_num_long = 0
    prev_num_short = 0
    entry_price = None
    entry_type = None

    for idx, row in history_df.iterrows():
        num_long = row.get('num_long', 0)
        num_short = row.get('num_short', 0)
        price = row.get('price', row.get('Close', 0))

        # Long entry
        if num_long > prev_num_long and entry_price is None:
            entry_price = price
            entry_type = 'long'

        # Long exit
        elif num_long < prev_num_long and entry_type == 'long':
            pnl = (price - entry_price) / entry_price
            trades.append(pnl)
            entry_price = None
            entry_type = None

        # Short entry
        if num_short > prev_num_short and entry_price is None:
            entry_price = price
            entry_type = 'short'

        # Short exit
        elif num_short < prev_num_short and entry_type == 'short':
            pnl = (entry_price - price) / entry_price
            trades.append(pnl)
            entry_price = None
            entry_type = None

        prev_num_long = num_long
        prev_num_short = num_short

    if len(trades) == 0:
        return {
            'win_rate': 0.0,
            'win_rate_pct': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }

    # Calculate win rate
    winning_trades = sum(1 for pnl in trades if pnl > 0)
    losing_trades = sum(1 for pnl in trades if pnl <= 0)
    win_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0

    return {
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'avg_win': np.mean([pnl for pnl in trades if pnl > 0]) if winning_trades > 0 else 0.0,
        'avg_loss': np.mean([pnl for pnl in trades if pnl <= 0]) if losing_trades > 0 else 0.0
    }


def calculate_all_metrics(
    portfolio_values: pd.Series,
    history_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760
) -> Dict:
    """
    Calculate all performance metrics.

    Args:
        portfolio_values: Series of portfolio values over time
        history_df: Portfolio history DataFrame
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Dictionary with all metrics
    """
    # Calculate returns
    returns = calculate_returns(portfolio_values)

    # Calculate all metrics
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(returns, portfolio_values, periods_per_year),
    }

    # Add drawdown info
    dd_info = calculate_max_drawdown(portfolio_values)
    metrics.update(dd_info)

    # Add win rate info
    wr_info = calculate_win_rate(history_df)
    metrics.update(wr_info)

    # Add return info
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    metrics['total_return'] = total_return
    metrics['total_return_pct'] = total_return * 100

    return metrics
