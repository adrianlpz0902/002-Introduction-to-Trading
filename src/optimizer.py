"""
Hyperparameter Optimization Module

Uses Optuna for Bayesian optimization to maximize Calmar Ratio.
"""

import pandas as pd
import numpy as np
import optuna
from typing import Dict, Optional
from .indicators import add_all_indicators
from .signals import create_signals
from .backtest import run_backtest
from .metrics import calculate_all_metrics


def objective_function(
    trial: optuna.Trial,
    train_data: pd.DataFrame,
    initial_cash: float = 10000.0,
    transaction_fee: float = 0.00125,
    price_col: str = 'close'
) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        train_data: Training data (without indicators)
        initial_cash: Starting cash
        transaction_fee: Transaction fee percentage
        price_col: Price column name

    Returns:
        Calmar Ratio (to be maximized)
    """
    # Suggest hyperparameters
    params = {
        # Indicator parameters
        'rsi_period': trial.suggest_int('rsi_period', 10, 20),
        'rsi_oversold': trial.suggest_int('rsi_oversold', 25, 35),
        'rsi_overbought': trial.suggest_int('rsi_overbought', 65, 75),
        'macd_fast': trial.suggest_int('macd_fast', 8, 15),
        'macd_slow': trial.suggest_int('macd_slow', 20, 30),
        'macd_signal': trial.suggest_int('macd_signal', 7, 11),
        'bb_period': trial.suggest_int('bb_period', 15, 25),
        'bb_std': trial.suggest_float('bb_std', 1.5, 2.5),

        # Trading parameters
        'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05),
        'take_profit': trial.suggest_float('take_profit', 0.02, 0.10),
    }

    try:
        # Add indicators with suggested parameters
        data_with_indicators = add_all_indicators(
            train_data.copy(),
            rsi_period=params['rsi_period'],
            macd_fast=params['macd_fast'],
            macd_slow=params['macd_slow'],
            macd_signal=params['macd_signal'],
            bb_period=params['bb_period'],
            bb_std=params['bb_std'],
            price_col=price_col
        )

        # Create signals
        data_with_signals = create_signals(
            data_with_indicators,
            rsi_oversold=params['rsi_oversold'],
            rsi_overbought=params['rsi_overbought'],
            min_confirmations=2,
            price_col=price_col
        )

        # Run backtest
        portfolio, history_df = run_backtest(
            data_with_signals,
            initial_cash=initial_cash,
            transaction_fee=transaction_fee,
            stop_loss=params['stop_loss'],
            take_profit=params['take_profit'],
            price_col=price_col
        )

        # Calculate metrics
        portfolio_values = history_df['portfolio_value']
        metrics = calculate_all_metrics(portfolio_values, history_df)

        # Return Calmar Ratio (objective to maximize)
        calmar = metrics['calmar_ratio']

        # Handle edge cases
        if np.isnan(calmar) or np.isinf(calmar):
            return -1000.0

        return calmar

    except Exception as e:
        # Return very poor score on error
        print(f"Trial failed: {e}")
        return -1000.0


def optimize_strategy(
    train_data: pd.DataFrame,
    n_trials: int = 100,
    initial_cash: float = 10000.0,
    transaction_fee: float = 0.00125,
    price_col: str = 'close',
    verbose: bool = True
) -> Dict:
    """
    Optimize strategy hyperparameters using Optuna.

    Args:
        train_data: Training data (without indicators)
        n_trials: Number of optimization trials
        initial_cash: Starting cash
        transaction_fee: Transaction fee percentage
        price_col: Price column name
        verbose: Print progress

    Returns:
        Dictionary with best parameters and best Calmar Ratio
    """
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )

    # Optimize
    study.optimize(
        lambda trial: objective_function(
            trial, train_data, initial_cash, transaction_fee, price_col
        ),
        n_trials=n_trials,
        show_progress_bar=verbose
    )

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    if verbose:
        print(f"\nOptimization Complete!")
        print(f"Best Calmar Ratio: {best_value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

    return {
        'best_params': best_params,
        'best_calmar_ratio': best_value,
        'study': study
    }


def walk_forward_analysis(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int,
    n_trials: int = 50,
    initial_cash: float = 10000.0,
    transaction_fee: float = 0.00125,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Perform walk-forward analysis to prevent overfitting.

    Args:
        data: Full dataset
        train_size: Size of training window
        test_size: Size of test window
        step_size: Step size for rolling window
        n_trials: Number of trials per optimization
        initial_cash: Starting cash
        transaction_fee: Transaction fee percentage
        price_col: Price column name

    Returns:
        DataFrame with walk-forward results
    """
    results = []

    # Perform rolling window optimization
    for start_idx in range(0, len(data) - train_size - test_size + 1, step_size):
        train_end = start_idx + train_size
        test_end = train_end + test_size

        # Split data
        train_data = data.iloc[start_idx:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()

        print(f"\nWindow {len(results)+1}: Train [{start_idx}:{train_end}], Test [{train_end}:{test_end}]")

        # Optimize on training data
        opt_result = optimize_strategy(
            train_data,
            n_trials=n_trials,
            initial_cash=initial_cash,
            transaction_fee=transaction_fee,
            price_col=price_col,
            verbose=False
        )

        best_params = opt_result['best_params']

        # Test on out-of-sample data
        data_with_indicators = add_all_indicators(
            test_data.copy(),
            rsi_period=best_params['rsi_period'],
            macd_fast=best_params['macd_fast'],
            macd_slow=best_params['macd_slow'],
            macd_signal=best_params['macd_signal'],
            bb_period=best_params['bb_period'],
            bb_std=best_params['bb_std'],
            price_col=price_col
        )

        data_with_signals = create_signals(
            data_with_indicators,
            rsi_oversold=best_params['rsi_oversold'],
            rsi_overbought=best_params['rsi_overbought'],
            price_col=price_col
        )

        portfolio, history_df = run_backtest(
            data_with_signals,
            initial_cash=initial_cash,
            transaction_fee=transaction_fee,
            stop_loss=best_params['stop_loss'],
            take_profit=best_params['take_profit'],
            price_col=price_col
        )

        # Calculate test metrics
        test_metrics = calculate_all_metrics(
            history_df['portfolio_value'],
            history_df
        )

        # Store results
        results.append({
            'window': len(results) + 1,
            'train_start': start_idx,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'train_calmar': opt_result['best_calmar_ratio'],
            'test_calmar': test_metrics['calmar_ratio'],
            'test_sharpe': test_metrics['sharpe_ratio'],
            'test_sortino': test_metrics['sortino_ratio'],
            'test_return': test_metrics['total_return_pct'],
            'test_max_dd': test_metrics['max_drawdown_pct'],
            'test_win_rate': test_metrics['win_rate_pct'],
            **{f'param_{k}': v for k, v in best_params.items()}
        })

        print(f"  Train Calmar: {opt_result['best_calmar_ratio']:.4f}")
        print(f"  Test Calmar:  {test_metrics['calmar_ratio']:.4f}")

    return pd.DataFrame(results)
