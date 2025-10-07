"""
Configuration File

Central configuration for all strategy parameters.
"""

# Data configuration
DATA_CONFIG = {
    'filepath': 'data/Binance_BTCUSDT_1h.csv',
    'train_ratio': 0.60,
    'test_ratio': 0.20,
    'val_ratio': 0.20,
    'price_col': 'Close'  # Changed to match CSV column name (capitalized)
}

# Best parameters from run_20251006_103414 (0.30 Calmar training)
STRATEGY_PARAMS = {
    # Technical Indicators
    'rsi_period': 37,
    'rsi_oversold': 20,
    'rsi_overbought': 73,

    'macd_fast': 8,
    'macd_slow': 27,
    'macd_signal': 9,

    'bb_period': 25,
    'bb_std': 2.319547964394184,

    # Trading Rules
    'stop_loss': 0.13638190484328916,  # 13.6%
    'take_profit': 0.13067770359847403,  # 13.1%
    'min_confirmations': 2  # 2-of-3 rule
}

# Portfolio configuration (simplified with risk management)
PORTFOLIO_CONFIG = {
    'initial_cash': 10000.0,
    'transaction_fee': 0.00125,  # 0.125%
    'position_size': 0.95,  # Use 95% of available cash per trade

    # Risk Management (prevents >100% drawdown)
    'max_positions': 1,  # Only one position at a time (prevents compounding losses)
    'max_position_size_pct': 0.30,  # Max 30% of initial capital per trade
    'min_portfolio_pct': 0.20,  # Circuit breaker: stop trading if portfolio < 20% of initial
    'max_drawdown_limit': 0.40  # Circuit breaker: stop trading if drawdown > 40%
}

# Optimization configuration
OPTIMIZATION_CONFIG = {
    'n_trials': 200,  # Increased from 100 for better hyperparameter exploration
    'objective': 'calmar_ratio',  # Primary objective to maximize
    'use_optimization': False,  # Using best parameters from run_103414

    # Hyperparameter search space (conservative to ensure trading activity)
    'search_space': {
        'rsi_period': (10, 30),  # Conservative range to avoid overly slow signals
        'rsi_oversold': (20, 35),  # Avoid extreme lows that rarely trigger
        'rsi_overbought': (65, 80),  # Avoid unreachable highs
        'macd_fast': (8, 15),
        'macd_slow': (20, 30),
        'macd_signal': (7, 11),
        'bb_period': (15, 25),
        'bb_std': (1.5, 2.5),
        'stop_loss': (0.02, 0.08),  # 2%-8% realistic risk management
        'take_profit': (0.03, 0.10)  # 3%-10% achievable profit targets
    }
}

# Walk-forward analysis configuration
WALKFORWARD_CONFIG = {
    'enabled': False,  # Disabled - using pre-validated best parameters from run_103414
    'train_size': 5000,  # Number of periods for training (larger window)
    'test_size': 1000,  # Number of periods for testing
    'step_size': 1000,  # Rolling window step size
    'n_trials': 50  # Trials per window
}

# Metrics configuration
METRICS_CONFIG = {
    'risk_free_rate': 0.0,  # Annual risk-free rate
    'periods_per_year': 8760  # Hourly data: 24 * 365
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'output_dir': 'results/figures',
    'save_plots': True,
    'show_plots': False,
    'dpi': 300
}

# Output configuration
OUTPUT_CONFIG = {
    'results_dir': 'results',
    'figures_dir': 'results/figures',
    'tables_dir': 'results/tables',
    'save_results': True,
    'use_timestamp': True  # Create timestamped folders to prevent overwriting
}
