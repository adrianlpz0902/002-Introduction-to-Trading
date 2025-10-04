# Project Structure - Trading Strategy 002

## 📁 Directory Tree

```
002-Introduction-to-Trading/
│
├── data/
│   └── Binance_BTCUSDT_1h.csv          # BTC/USDT hourly OHLCV data
│
├── src/                                 # Core strategy modules
│   ├── __init__.py                      # Package initialization
│   ├── data_loader.py                   # Data loading & 60/20/20 split
│   ├── indicators.py                    # RSI, MACD, Bollinger Bands
│   ├── signals.py                       # 2-of-3 signal confirmation
│   ├── backtest.py                      # Backtesting engine
│   ├── portfolio.py                     # Portfolio & position management
│   ├── metrics.py                       # Performance metrics
│   ├── optimizer.py                     # Optuna hyperparameter optimization
│   └── visualization.py                 # Charts & tables generation
│
├── notebooks/
│   └── exploration.ipynb                # Interactive data exploration
│
├── results/
│   ├── figures/                         # Generated charts (auto-created)
│   └── tables/                          # Performance tables (auto-created)
│
├── tests/                               # Unit tests
│   ├── __init__.py
│   ├── test_indicators.py               # Test technical indicators
│   ├── test_signals.py                  # Test signal generation
│   ├── test_backtest.py                 # Test backtesting engine
│   └── test_metrics.py                  # Test performance metrics
│
├── main.py                              # Main execution pipeline
├── config.py                            # Configuration parameters
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
├── README.md                            # Project documentation
├── project_requirements.md              # Requirements specification
└── PROJECT_STRUCTURE.md                 # This file
```

## 🎯 Quick Start Guide

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python main.py
```

This will:
- Load and split data (60% train, 20% test, 20% validation)
- Run hyperparameter optimization (100 trials)
- Backtest on all datasets
- Calculate performance metrics
- Generate visualizations

### 3. Explore Data Interactively

```bash
jupyter notebook notebooks/exploration.ipynb
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## 📊 Module Overview

### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **data_loader.py** | Load & preprocess data | `load_and_preprocess_data()`, `clean_data()` |
| **indicators.py** | Technical indicators | `calculate_rsi()`, `calculate_macd()`, `calculate_bollinger_bands()` |
| **signals.py** | Signal generation | `create_signals()`, `apply_confirmation_logic()` |
| **backtest.py** | Trading simulation | `run_backtest()` |
| **portfolio.py** | Portfolio management | `Portfolio` class with long/short positions |
| **metrics.py** | Performance analysis | `calculate_sharpe_ratio()`, `calculate_calmar_ratio()`, etc. |
| **optimizer.py** | Hyperparameter tuning | `optimize_strategy()`, `walk_forward_analysis()` |
| **visualization.py** | Charts & tables | `generate_all_visualizations()` |

### Configuration (config.py)

Edit `config.py` to customize:
- Data split ratios
- Strategy parameters (RSI periods, MACD settings, etc.)
- Optimization settings (number of trials, search space)
- Portfolio settings (initial cash, transaction fees)
- Visualization preferences

### Main Pipeline (main.py)

The main pipeline orchestrates:
1. Data loading and preprocessing
2. Hyperparameter optimization (optional)
3. Walk-forward analysis (optional)
4. Backtesting on train/test/validation
5. Metrics calculation
6. Visualization generation
7. Results export

## 🎨 Output Files

After running `main.py`, the following files are generated:

### Figures (`results/figures/`)
- `train_portfolio_value.png` - Portfolio value over time
- `train_cumulative_returns.png` - Cumulative returns
- `train_drawdown.png` - Drawdown analysis
- `train_monthly_returns_heatmap.png` - Monthly returns heatmap
- (Same for `test_` and `validation_` prefixes)

### Tables (`results/tables/`)
- `summary_metrics.csv` - Performance summary across all datasets
- `best_parameters.csv` - Optimized hyperparameters
- `train_performance_metrics.csv` - Detailed training metrics
- `train_history.csv` - Complete portfolio history
- (Same for test and validation)

## 🧪 Testing

Comprehensive unit tests ensure code correctness:

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_indicators.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Customization

### Modify Strategy Parameters

Edit `config.py`:

```python
STRATEGY_PARAMS = {
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'stop_loss': 0.02,
    'take_profit': 0.03,
    # ...
}
```

### Change Optimization Settings

```python
OPTIMIZATION_CONFIG = {
    'n_trials': 100,  # Increase for more thorough optimization
    'use_optimization': True,  # Set to False to skip
}
```

### Enable Walk-Forward Analysis

```python
WALKFORWARD_CONFIG = {
    'enabled': True,  # Enable walk-forward validation
    'train_size': 1000,
    'test_size': 200,
    'step_size': 100,
}
```

## 📈 Strategy Logic

### 2-of-3 Signal Confirmation

A position is only opened when **at least 2 out of 3 indicators agree**:

**Buy Signal:**
- RSI < 30 (oversold)
- MACD crosses above signal line
- Price touches lower Bollinger Band

**Sell Signal:**
- RSI > 70 (overbought)
- MACD crosses below signal line
- Price touches upper Bollinger Band

### Position Management
- **Long positions**: Opened on buy signals
- **Short positions**: Opened on sell signals
- **Exit conditions**: Opposite signal, stop loss, or take profit
- **Transaction fees**: 0.125% on entry and exit

## 📝 Performance Metrics

The system calculates:
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Return vs. downside volatility
- **Calmar Ratio**: Return vs. maximum drawdown (primary objective)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## 🚀 Next Steps

1. Run the pipeline: `python main.py`
2. Review results in `results/` directory
3. Analyze performance metrics
4. Iterate on strategy parameters
5. Document findings in executive report

## 📚 Additional Notes

- All code is modular and well-documented
- Each function has docstrings explaining parameters and returns
- Test coverage ensures reliability
- Git-friendly structure for version control
- Ready for academic submission and presentation
