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
    'price_col': 'close'
}

# Initial trading parameters (before optimization)
STRATEGY_PARAMS = {
    # Technical Indicators
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,

    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,

    'bb_period': 20,
    'bb_std': 2.0,

    # Trading Rules
    'stop_loss': 0.02,  # 2%
    'take_profit': 0.03,  # 3%
    'min_confirmations': 2  # 2-of-3 rule
}

# Portfolio configuration
PORTFOLIO_CONFIG = {
    'initial_cash': 10000.0,
    'transaction_fee': 0.00125,  # 0.125%
    'position_size': None  # None = use all available cash
}

# Optimization configuration
OPTIMIZATION_CONFIG = {
    'n_trials': 100,
    'objective': 'calmar_ratio',  # Primary objective to maximize
    'use_optimization': True,  # Set to False to skip optimization

    # Hyperparameter search space
    'search_space': {
        'rsi_period': (10, 20),
        'rsi_oversold': (25, 35),
        'rsi_overbought': (65, 75),
        'macd_fast': (8, 15),
        'macd_slow': (20, 30),
        'macd_signal': (7, 11),
        'bb_period': (15, 25),
        'bb_std': (1.5, 2.5),
        'stop_loss': (0.01, 0.05),
        'take_profit': (0.02, 0.10)
    }
}

# Walk-forward analysis configuration
WALKFORWARD_CONFIG = {
    'enabled': False,  # Enable for production use
    'train_size': 1000,  # Number of periods for training
    'test_size': 200,  # Number of periods for testing
    'step_size': 100,  # Rolling window step size
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
    'save_results': True
}
