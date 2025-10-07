"""
Visualization Module

Generates charts and performance tables for strategy analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
import os


def plot_portfolio_value(
    history_df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Portfolio Value Over Time"
):
    """
    Plot portfolio value over time.

    Args:
        history_df: Portfolio history DataFrame
        save_path: Path to save figure (if None, displays plot)
        title: Plot title
    """
    plt.figure(figsize=(14, 6))

    plt.plot(history_df.index, history_df['portfolio_value'], linewidth=2, label='Portfolio Value')

    # Mark buy/sell signals
    buy_signals = history_df[history_df['signal'] == 'buy']
    sell_signals = history_df[history_df['signal'] == 'sell']

    if len(buy_signals) > 0:
        plt.scatter(buy_signals.index, buy_signals['portfolio_value'],
                   marker='^', color='green', s=100, label='Buy Signal', zorder=5)

    if len(sell_signals) > 0:
        plt.scatter(sell_signals.index, sell_signals['portfolio_value'],
                   marker='v', color='red', s=100, label='Sell Signal', zorder=5)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_returns(
    history_df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Cumulative Returns"
):
    """
    Plot cumulative returns.

    Args:
        history_df: Portfolio history DataFrame
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(14, 6))

    # Calculate cumulative returns
    initial_value = history_df['portfolio_value'].iloc[0]
    cumulative_returns = (history_df['portfolio_value'] / initial_value - 1) * 100

    plt.plot(history_df.index, cumulative_returns, linewidth=2, color='steelblue')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_drawdown(
    history_df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Drawdown Analysis"
):
    """
    Plot drawdown over time.

    Args:
        history_df: Portfolio history DataFrame
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(14, 6))

    # Calculate drawdown
    portfolio_values = history_df['portfolio_value']
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max * 100

    plt.fill_between(history_df.index, drawdown, 0, color='red', alpha=0.3)
    plt.plot(history_df.index, drawdown, color='darkred', linewidth=2)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_monthly_returns_heatmap(
    history_df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Monthly Returns Heatmap"
):
    """
    Create heatmap of monthly returns.

    Args:
        history_df: Portfolio history DataFrame with timestamp
        save_path: Path to save figure
        title: Plot title
    """
    # Ensure we have timestamp
    if 'timestamp' not in history_df.columns:
        print("Warning: No timestamp column found. Skipping monthly returns heatmap.")
        return

    # Set timestamp as index if not already
    df = history_df.copy()
    if df.index.name != 'timestamp':
        # Try to convert timestamp to datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        except:
            print("Warning: Could not convert timestamp to datetime. Skipping monthly returns heatmap.")
            return

    # Calculate daily returns
    df['returns'] = df['portfolio_value'].pct_change()

    # Resample to monthly returns
    monthly_returns = df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

    # Create year-month pivot table
    monthly_returns_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })

    pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='return')

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'Return (%)'})

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def create_performance_table(metrics: Dict) -> pd.DataFrame:
    """
    Create a formatted performance metrics table.

    Args:
        metrics: Dictionary with performance metrics

    Returns:
        DataFrame with formatted metrics
    """
    table_data = {
        'Metric': [
            'Total Return (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Max Drawdown (%)',
            'Win Rate (%)',
            'Total Trades',
            'Winning Trades',
            'Losing Trades',
            'Avg Win (%)',
            'Avg Loss (%)'
        ],
        'Value': [
            f"{metrics.get('total_return_pct', 0):.2f}",
            f"{metrics.get('sharpe_ratio', 0):.4f}",
            f"{metrics.get('sortino_ratio', 0):.4f}",
            f"{metrics.get('calmar_ratio', 0):.4f}",
            f"{metrics.get('max_drawdown_pct', 0):.2f}",
            f"{metrics.get('win_rate_pct', 0):.2f}",
            f"{metrics.get('total_trades', 0)}",
            f"{metrics.get('winning_trades', 0)}",
            f"{metrics.get('losing_trades', 0)}",
            f"{metrics.get('avg_win', 0) * 100:.2f}",
            f"{metrics.get('avg_loss', 0) * 100:.2f}"
        ]
    }

    return pd.DataFrame(table_data)


def generate_all_visualizations(
    history_df: pd.DataFrame,
    metrics: Dict,
    output_dir: str = 'results/figures',
    prefix: str = ''
):
    """
    Generate all visualizations and save to output directory.

    Args:
        history_df: Portfolio history DataFrame
        metrics: Performance metrics dictionary
        output_dir: Directory to save figures
        prefix: Prefix for filenames (e.g., 'train_', 'test_', 'val_')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating visualizations in {output_dir}...")

    # Generate all plots
    plot_portfolio_value(
        history_df,
        save_path=os.path.join(output_dir, f'{prefix}portfolio_value.png'),
        title=f"{prefix.replace('_', ' ').title()}Portfolio Value Over Time"
    )

    plot_returns(
        history_df,
        save_path=os.path.join(output_dir, f'{prefix}cumulative_returns.png'),
        title=f"{prefix.replace('_', ' ').title()}Cumulative Returns"
    )

    plot_drawdown(
        history_df,
        save_path=os.path.join(output_dir, f'{prefix}drawdown.png'),
        title=f"{prefix.replace('_', ' ').title()}Drawdown Analysis"
    )

    plot_monthly_returns_heatmap(
        history_df,
        save_path=os.path.join(output_dir, f'{prefix}monthly_returns_heatmap.png'),
        title=f"{prefix.replace('_', ' ').title()}Monthly Returns Heatmap"
    )

    # Save performance table
    perf_table = create_performance_table(metrics)
    table_path = os.path.join(output_dir.replace('figures', 'tables'), f'{prefix}performance_metrics.csv')
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    perf_table.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")

    print(f"\nAll visualizations generated successfully!")
    print(f"\nPerformance Summary:")
    print(perf_table.to_string(index=False))
