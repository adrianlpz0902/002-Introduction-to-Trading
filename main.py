"""
Main Execution Script

Orchestrates the complete trading strategy pipeline:
1. Load and preprocess data
2. Add technical indicators
3. Generate trading signals
4. Optimize hyperparameters (optional)
5. Run backtesting on train/test/validation sets
6. Calculate performance metrics
7. Generate visualizations
"""

import os
import pandas as pd
from datetime import datetime

# Import modules
from src.data_loader import load_and_preprocess_data
from src.indicators import add_all_indicators
from src.signals import create_signals
from src.backtest_simple import run_backtest  # Using simplified backtest
from src.metrics import calculate_all_metrics
from src.optimizer import optimize_strategy, walk_forward_analysis
from src.visualization import generate_all_visualizations, create_performance_table

# Import configuration
from config import (
    DATA_CONFIG,
    STRATEGY_PARAMS,
    PORTFOLIO_CONFIG,
    OPTIMIZATION_CONFIG,
    WALKFORWARD_CONFIG,
    METRICS_CONFIG,
    VISUALIZATION_CONFIG,
    OUTPUT_CONFIG
)


def run_strategy_pipeline(optimize: bool = True, verbose: bool = True):
    """
    Run the complete trading strategy pipeline.

    Args:
        optimize: Whether to run hyperparameter optimization
        verbose: Print detailed progress
    """
    print("=" * 80)
    print("TRADING STRATEGY PROJECT 002")
    print("Multi-Indicator Strategy with 2-of-3 Confirmation")
    print("=" * 80)

    # Create timestamped output directories if enabled
    if OUTPUT_CONFIG.get('use_timestamp', True):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_results_dir = OUTPUT_CONFIG['results_dir']
        run_dir = os.path.join(base_results_dir, f'run_{timestamp}')

        # Update output paths for this run
        OUTPUT_CONFIG['figures_dir'] = os.path.join(run_dir, 'figures')
        OUTPUT_CONFIG['tables_dir'] = os.path.join(run_dir, 'tables')
        OUTPUT_CONFIG['results_dir'] = run_dir

        print(f"\nResults will be saved to: {run_dir}/")
    else:
        print(f"\nResults will be saved to: {OUTPUT_CONFIG['results_dir']}/")
        print("WARNING: Existing results will be overwritten!")

    # ========================================
    # 1. LOAD AND PREPROCESS DATA
    # ========================================
    print("\n[1/7] Loading and preprocessing data...")
    train_data, test_data, val_data = load_and_preprocess_data(
        filepath=DATA_CONFIG['filepath'],
        train_ratio=DATA_CONFIG['train_ratio'],
        test_ratio=DATA_CONFIG['test_ratio'],
        val_ratio=DATA_CONFIG['val_ratio']
    )

    # ========================================
    # 2. HYPERPARAMETER OPTIMIZATION
    # ========================================
    best_params = STRATEGY_PARAMS.copy()

    if optimize and OPTIMIZATION_CONFIG['use_optimization']:
        print(f"\n[2/7] Running hyperparameter optimization ({OPTIMIZATION_CONFIG['n_trials']} trials)...")
        print("This may take several minutes...")

        opt_result = optimize_strategy(
            train_data=train_data,
            n_trials=OPTIMIZATION_CONFIG['n_trials'],
            initial_cash=PORTFOLIO_CONFIG['initial_cash'],
            transaction_fee=PORTFOLIO_CONFIG['transaction_fee'],
            price_col=DATA_CONFIG['price_col'],
            verbose=verbose
        )

        best_params.update(opt_result['best_params'])
        print(f"\nOptimization complete! Best Calmar Ratio: {opt_result['best_calmar_ratio']:.4f}")
    else:
        print("\n[2/7] Skipping optimization (using default parameters)...")

    # ========================================
    # 3. WALK-FORWARD ANALYSIS (OPTIONAL)
    # ========================================
    if WALKFORWARD_CONFIG['enabled']:
        print("\n[3/7] Running walk-forward analysis...")
        wf_results = walk_forward_analysis(
            data=pd.concat([train_data, test_data]),
            train_size=WALKFORWARD_CONFIG['train_size'],
            test_size=WALKFORWARD_CONFIG['test_size'],
            step_size=WALKFORWARD_CONFIG['step_size'],
            n_trials=WALKFORWARD_CONFIG['n_trials'],
            initial_cash=PORTFOLIO_CONFIG['initial_cash'],
            transaction_fee=PORTFOLIO_CONFIG['transaction_fee'],
            price_col=DATA_CONFIG['price_col']
        )

        # Save walk-forward results
        wf_path = os.path.join(OUTPUT_CONFIG['tables_dir'], 'walk_forward_results.csv')
        os.makedirs(OUTPUT_CONFIG['tables_dir'], exist_ok=True)
        wf_results.to_csv(wf_path, index=False)
        print(f"Walk-forward results saved to {wf_path}")
    else:
        print("\n[3/7] Skipping walk-forward analysis...")

    # ========================================
    # 4. RUN BACKTESTING ON ALL DATASETS
    # ========================================
    print("\n[4/7] Running backtesting on all datasets...")

    results = {}

    for dataset_name, dataset in [('train', train_data), ('test', test_data), ('validation', val_data)]:
        print(f"\n  Processing {dataset_name} set...")

        # Add indicators
        data_with_indicators = add_all_indicators(
            dataset.copy(),
            rsi_period=best_params['rsi_period'],
            macd_fast=best_params['macd_fast'],
            macd_slow=best_params['macd_slow'],
            macd_signal=best_params.get('macd_signal', STRATEGY_PARAMS['macd_signal']),
            bb_period=best_params['bb_period'],
            bb_std=best_params['bb_std'],
            price_col=DATA_CONFIG['price_col']
        )

        # Create signals
        data_with_signals = create_signals(
            data_with_indicators,
            rsi_oversold=best_params['rsi_oversold'],
            rsi_overbought=best_params['rsi_overbought'],
            min_confirmations=STRATEGY_PARAMS['min_confirmations'],
            price_col=DATA_CONFIG['price_col']
        )

        # Run simplified backtest with risk management
        history_df, portfolio = run_backtest(
            data_with_signals,
            initial_cash=PORTFOLIO_CONFIG['initial_cash'],
            transaction_fee=PORTFOLIO_CONFIG['transaction_fee'],
            stop_loss=best_params['stop_loss'],
            take_profit=best_params['take_profit'],
            position_size_pct=PORTFOLIO_CONFIG['position_size'],
            price_col=DATA_CONFIG['price_col'],
            max_positions=PORTFOLIO_CONFIG['max_positions'],
            max_position_size_pct=PORTFOLIO_CONFIG['max_position_size_pct'],
            min_portfolio_pct=PORTFOLIO_CONFIG['min_portfolio_pct'],
            max_drawdown_limit=PORTFOLIO_CONFIG['max_drawdown_limit']
        )

        # Calculate metrics
        metrics = calculate_all_metrics(
            portfolio_values=history_df['portfolio_value'],
            history_df=history_df,
            risk_free_rate=METRICS_CONFIG['risk_free_rate'],
            periods_per_year=METRICS_CONFIG['periods_per_year']
        )

        # Simple diagnostics
        print(f"\n  📊 {dataset_name.upper()} RESULTS:")
        print(f"    Portfolio: ${PORTFOLIO_CONFIG['initial_cash']:.2f} → ${history_df['portfolio_value'].iloc[-1]:.2f}")
        print(f"    Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        print(f"    Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"    Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"    Total Trades: {metrics['total_trades']}")
        print(f"    Win Rate: {metrics['win_rate_pct']:.2f}%")

        # Store results
        results[dataset_name] = {
            'portfolio': portfolio,
            'history_df': history_df,
            'metrics': metrics,
            'data_with_signals': data_with_signals
        }

    # ========================================
    # 5. CALCULATE METRICS
    # ========================================
    print("\n[5/7] Calculating performance metrics...")
    print("\nPerformance Summary:")
    print("-" * 80)

    summary_table = pd.DataFrame({
        'Metric': ['Calmar Ratio', 'Sharpe Ratio', 'Sortino Ratio', 'Total Return (%)',
                   'Max Drawdown (%)', 'Win Rate (%)', 'Total Trades'],
        'Training': [
            f"{results['train']['metrics']['calmar_ratio']:.4f}",
            f"{results['train']['metrics']['sharpe_ratio']:.4f}",
            f"{results['train']['metrics']['sortino_ratio']:.4f}",
            f"{results['train']['metrics']['total_return_pct']:.2f}",
            f"{results['train']['metrics']['max_drawdown_pct']:.2f}",
            f"{results['train']['metrics']['win_rate_pct']:.2f}",
            f"{results['train']['metrics']['total_trades']}"
        ],
        'Test': [
            f"{results['test']['metrics']['calmar_ratio']:.4f}",
            f"{results['test']['metrics']['sharpe_ratio']:.4f}",
            f"{results['test']['metrics']['sortino_ratio']:.4f}",
            f"{results['test']['metrics']['total_return_pct']:.2f}",
            f"{results['test']['metrics']['max_drawdown_pct']:.2f}",
            f"{results['test']['metrics']['win_rate_pct']:.2f}",
            f"{results['test']['metrics']['total_trades']}"
        ],
        'Validation': [
            f"{results['validation']['metrics']['calmar_ratio']:.4f}",
            f"{results['validation']['metrics']['sharpe_ratio']:.4f}",
            f"{results['validation']['metrics']['sortino_ratio']:.4f}",
            f"{results['validation']['metrics']['total_return_pct']:.2f}",
            f"{results['validation']['metrics']['max_drawdown_pct']:.2f}",
            f"{results['validation']['metrics']['win_rate_pct']:.2f}",
            f"{results['validation']['metrics']['total_trades']}"
        ]
    })

    print(summary_table.to_string(index=False))

    # ========================================
    # 6. GENERATE VISUALIZATIONS
    # ========================================
    print("\n[6/7] Generating visualizations...")

    for dataset_name in ['train', 'test', 'validation']:
        generate_all_visualizations(
            history_df=results[dataset_name]['history_df'],
            metrics=results[dataset_name]['metrics'],
            output_dir=OUTPUT_CONFIG['figures_dir'],
            prefix=f'{dataset_name}_'
        )

    # ========================================
    # 7. SAVE RESULTS
    # ========================================
    print("\n[7/7] Saving results...")

    # Save summary table
    summary_path = os.path.join(OUTPUT_CONFIG['tables_dir'], 'summary_metrics.csv')
    os.makedirs(OUTPUT_CONFIG['tables_dir'], exist_ok=True)
    summary_table.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # Save best parameters
    params_df = pd.DataFrame([best_params])
    params_path = os.path.join(OUTPUT_CONFIG['tables_dir'], 'best_parameters.csv')
    params_df.to_csv(params_path, index=False)
    print(f"Best parameters saved to {params_path}")

    # Save detailed history for each dataset
    for dataset_name in ['train', 'test', 'validation']:
        history_path = os.path.join(OUTPUT_CONFIG['tables_dir'], f'{dataset_name}_history.csv')
        results[dataset_name]['history_df'].to_csv(history_path, index=False)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_CONFIG['results_dir']}/")
    print(f"  - Figures: {OUTPUT_CONFIG['figures_dir']}/")
    print(f"  - Tables: {OUTPUT_CONFIG['tables_dir']}/")

    return results, best_params


if __name__ == "__main__":
    # Run the complete pipeline
    results, best_params = run_strategy_pipeline(
        optimize=OPTIMIZATION_CONFIG['use_optimization'],
        verbose=True
    )

    print("\n✓ Strategy execution completed successfully!")
