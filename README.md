# 002-Introduction-to-Trading
Trading Strategy Project 002

A systematic algorithmic trading strategy implementation with multi-indicator signal confirmation, optimized for maximizing the Calmar Ratio.

 Project Overview
This project implements a multi-indicator technical trading strategy for BTC/USDT with the following key features:

3 Technical Indicators with 2-of-3 signal confirmation
Long & Short Positions (no leverage)
Realistic Backtesting with 0.125% transaction fees
Hyperparameter Optimization using Optuna (Bayesian optimization)
Walk-Forward Analysis to prevent overfitting
Comprehensive Performance Metrics and visualizations

Primary Objective: Maximize Calmar Ratio

 Key Results
<!-- Update these after running your strategy -->
MetricTrainingTestValidationCalmar RatioTBDTBDTBDSharpe RatioTBDTBDTBDSortino RatioTBDTBDTBDMax DrawdownTBDTBDTBDWin RateTBDTBDTBD

 Architecture
Modular Design
The project follows a modular architecture with clear separation of concerns:
project_002/
├── data/
│   └── Binance_BTCUSDT_1h.csv          # Raw OHLCV data
├── src/
│   ├── data_loader.py                   # Data loading & preprocessing
│   ├── indicators.py                    # Technical indicator calculations
│   ├── signals.py                       # Signal generation & confirmation
│   ├── backtest.py                      # Backtesting engine
│   ├── portfolio.py                     # Portfolio management
│   ├── metrics.py                       # Performance metrics
│   ├── optimizer.py                     # Hyperparameter optimization
│   └── visualization.py                 # Charts & tables generation
├── notebooks/
│   └── exploration.ipynb                # Data exploration
├── results/
│   ├── figures/                         # Generated charts
│   └── tables/                          # Performance tables
├── tests/
│   └── test_*.py                        # Unit tests
├── main.py                              # Main execution script
├── requirements.txt                     # Python dependencies
├── PROJECT_REQUIREMENTS.md              # Detailed requirements
└── README.md                            # This file
Data Flow
1. Raw OHLCV Data (Binance_BTCUSDT_1h.csv)
          ↓
2. Data Preprocessing & Splitting (60/20/20)
          ↓
3. Technical Indicators Calculation
          ↓
4. Signal Generation (2-of-3 Confirmation)
          ↓
5. Backtesting Engine (with transaction costs)
          ↓
6. Performance Metrics Calculation
          ↓
7. Hyperparameter Optimization (Optuna)
          ↓
8. Final Evaluation & Visualization

 Installation
Prerequisites

Python 3.8+
pip or conda

Setup

Clone the repository

bashgit clone <repository-url>
cd project_002

Create virtual environment

bashpython -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt
Required Libraries
txtpandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
ta>=0.11.0
optuna>=3.0.0
scipy>=1.9.0
plotly>=5.11.0
jupyter>=1.0.0

 Usage
Quick Start
Run the complete pipeline:
bashpython main.py
Step-by-Step Execution
pythonfrom src.data_loader import load_and_preprocess_data
from src.signals import create_signals
from src.backtest import run_backtest
from src.metrics import calculate_metrics
from src.optimizer import optimize_strategy
from src.visualization import generate_summary

# 1. Load data
train_data, test_data, val_data = load_and_preprocess_data('data/Binance_BTCUSDT_1h.csv')

# 2. Create signals
train_data_with_signals = create_signals(train_data, params)

# 3. Run backtesting
portfolio_hist, returns = run_backtest(train_data_with_signals, stop_loss, take_profit, n_shares)

# 4. Calculate metrics
metrics = calculate_metrics(portfolio_hist)

# 5. Optimize hyperparameters
best_params = optimize_strategy(train_data, n_trials=100)

# 6. Generate visualizations
generate_summary(portfolio_hist, returns, metrics)
Configuration
Edit parameters in main.py or create a config.yaml:
pythonSTRATEGY_PARAMS = {
    'indicators': ['RSI', 'MACD', 'BB'],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'bb_period': 20,
    'stop_loss': 0.02,
    'take_profit': 0.03,
    'n_shares': 1.0
}

DATA_SPLIT = {
    'train': 0.60,
    'test': 0.20,
    'validation': 0.20
}

OPTIMIZATION = {
    'n_trials': 100,
    'method': 'optuna',  # 'grid', 'random', or 'optuna'
    'objective': 'calmar_ratio'
}

Strategy Description
Technical Indicators

RSI (Relative Strength Index)

Period: 14 (optimized)
Buy signal: RSI < 30 (oversold)
Sell signal: RSI > 70 (overbought)


MACD (Moving Average Convergence Divergence)

Fast period: 12, Slow period: 26, Signal: 9
Buy signal: MACD crosses above signal line
Sell signal: MACD crosses below signal line


Bollinger Bands

Period: 20, Standard deviation: 2
Buy signal: Price touches lower band
Sell signal: Price touches upper band



Signal Confirmation Logic
2-of-3 Rule: A position is only opened when at least 2 out of 3 indicators agree.
pythonBUY_SIGNAL = (RSI_buy + MACD_buy + BB_buy) >= 2
SELL_SIGNAL = (RSI_sell + MACD_sell + BB_sell) >= 2
Position Management

Long positions: Opened on BUY_SIGNAL
Short positions: Opened on SELL_SIGNAL
Exit: Opposite signal or stop-loss/take-profit triggers
Transaction fees: 0.125% applied on entry and exit


 Methodology
Dataset Split

Training (60%): Used for hyperparameter optimization
Test (20%): Strategy validation during development
Validation (20%): Final unbiased performance evaluation

Walk-Forward Analysis
Implemented to prevent overfitting:

Optimize on training window
Test on out-of-sample data
Roll window forward
Repeat process

Optimization Process
Using Optuna (Bayesian optimization):
Optimized Parameters:

Stop loss range: [0.01, 0.05]
Take profit range: [0.02, 0.10]
RSI period: [10, 20]
MACD fast/slow: [8-15] / [20-30]
Bollinger period: [15, 25]

Objective Function: Maximize Calmar Ratio on training data

 Results & Analysis
Performance Summary
<!-- Add your results here after running -->
Best Strategy Performance (Validation Set):

Total Return: TBD%
Calmar Ratio: TBD
Sharpe Ratio: TBD
Sortino Ratio: TBD
Maximum Drawdown: TBD%
Win Rate: TBD%
Number of Trades: TBD

Visualizations
All charts are automatically generated in results/figures/:

portfolio_value.png - Portfolio value through time
returns_plot.png - Cumulative returns
drawdown_chart.png - Drawdown analysis
monthly_returns_heatmap.png - Monthly returns heatmap

Risk Analysis
Strengths:

TBD

Limitations:

TBD

Market Conditions:

Works best in: TBD
Struggles during: TBD


 Testing
Run unit tests:
bashpytest tests/
Run specific test module:
bashpytest tests/test_backtest.py -v

Module Documentation
1. data_loader.py
Handles data loading, cleaning, and train/test/validation splitting.
Key Functions:

load_and_preprocess_data() - Load CSV and split data
clean_data() - Handle missing values and outliers

2. indicators.py
Calculates technical indicators.
Key Functions:

calculate_rsi() - Relative Strength Index
calculate_macd() - MACD indicator
calculate_bollinger_bands() - Bollinger Bands

3. signals.py
Generates trading signals with 2-of-3 confirmation.
Key Functions:

create_signals() - Main signal generation function
apply_confirmation_logic() - 2-of-3 confirmation rule

4. backtest.py
Simulates trading strategy with realistic constraints.
Key Functions:

run_backtest() - Execute backtesting simulation
apply_transaction_costs() - Apply 0.125% fees

5. portfolio.py
Manages portfolio state and position tracking.
Key Functions:

update_position() - Update portfolio state
calculate_portfolio_value() - Track portfolio value

6. metrics.py
Calculates performance metrics.
Key Functions:

calculate_sharpe_ratio()
calculate_sortino_ratio()
calculate_calmar_ratio()
calculate_max_drawdown()
calculate_win_rate()

7. optimizer.py
Hyperparameter optimization using Optuna.
Key Functions:

optimize_strategy() - Main optimization loop
objective_function() - Optuna objective (Calmar)

8. visualization.py
Generates all required charts and tables.
Key Functions:

plot_portfolio_value()
plot_returns()
plot_drawdown()
create_returns_tables() - Monthly/Quarterly/Annual

 Contributing
This is an academic project.The developer of this projetct must:

Understand every line of code
Be able to explain implementation choices
Contribute meaningful commits to Git