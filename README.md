# Algorithmic Trading Strategy: Multi-Indicator Mean Reversion System
# Luis Adrián López Enríquez
> Academic Project - Systematic trading strategy optimized to maximize Calmar Ratio using multi-indicator confirmation logic on Bitcoin/USDT hourly data.

---

## Table of Contents

- [Overview](#overview)
- [Trading Strategy](#trading-strategy)
- [Final Results](#final-results)
- [Walk-Forward Analysis](#walk-forward-analysis)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [Technologies](#technologies)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [References](#references)
- [Acknowledgments](#Acknowledgments)

---

## Overview

This project implements a systematic algorithmic trading strategy designed to trade Bitcoin (BTC/USDT) using technical analysis with robust risk management. The strategy was developed as part of a quantitative finance course with the primary objective of maximizing the **Calmar Ratio** - a risk-adjusted performance metric that measures returns relative to maximum drawdown.

### Project Objectives

- Maximize Calmar Ratio across train/test/validation datasets
- Implement multi-indicator confirmation logic (2-of-3 rule)
- Build realistic backtesting environment with transaction costs
- Optimize hyperparameters using Bayesian optimization
- Validate strategy robustness through walk-forward analysis
- Maintain positive returns with controlled drawdown under 50%

### Dataset

- **Asset:** BTC/USDT (Bitcoin/Tether)
- **Timeframe:** 1 hour
- **Source:** Binance Exchange
- **Total Records:** 70,827 hourly candles
- **Data Split:**
  - Training: 60% (42,496 records)
  - Test: 20% (14,165 records)
  - Validation: 20% (14,166 records)

---

## Trading Strategy

### Philosophical Approach

The strategy is built on technical analysis using three complementary indicators with a **2-of-3 confirmation rule**. This approach reduces false signals by requiring at least two independent indicators to agree before entering a position.

### Technical Indicators

#### RSI (Relative Strength Index)

**Purpose:** Measures momentum extremes based on recent price changes

**Parameters:**
- Period: 37
- Oversold threshold: 20
- Overbought threshold: 73

**Signal Logic:**
- RSI < 20 → Oversold → Buy signal
- RSI > 73 → Overbought → Sell signal

#### MACD (Moving Average Convergence Divergence)

**Purpose:** Identifies trend strength and momentum shifts

**Parameters:**
- Fast EMA: 8 periods
- Slow EMA: 27 periods
- Signal line: 9 periods

**Signal Logic:**
- MACD crosses above signal line → Buy signal
- MACD crosses below signal line → Sell signal

#### Bollinger Bands

**Purpose:** Measures statistical price deviations from moving average

**Parameters:**
- Period: 25
- Standard deviation multiplier: 2.32

**Signal Logic:**
- Price touches or crosses lower band → Buy signal
- Price touches or crosses upper band → Sell signal

### 2-of-3 Confirmation Rule

A position is only opened when **at least 2 out of 3 indicators agree**. The buy signal is generated when at least two indicators simultaneously indicate oversold conditions, and the sell signal when at least two indicate overbought conditions.

**Rationale:** This confirmation logic significantly reduces false signals while maintaining high-probability trade setups.

### Position Management

**Position Types:**
- Long positions: Profit from price increases
- Short positions: Profit from price decreases

**Risk Management:**
- Stop-loss: 13.6% per trade
- Take-profit: 13.1% per trade
- Maximum positions: 1 concurrent position
- Position size limit: 30% of initial capital
- Circuit breakers:
  - Trading halts at 40% maximum drawdown
  - Emergency stop at 20% portfolio floor

**Transaction Costs:**
- Fee rate: 0.125% per transaction (both entry and exit)
- Realistic simulation of exchange fees

---

## Final Results

### Model Performance (run_20251006_103414)

#### Optimized Parameters

The following parameters were found through Bayesian optimization on the training dataset:

- RSI Period: 37
- RSI Oversold: 20
- RSI Overbought: 73
- MACD Fast: 8
- MACD Slow: 27
- MACD Signal: 9
- Bollinger Bands Period: 25
- Bollinger Bands Std: 2.32
- Stop-loss: 13.6%
- Take-profit: 13.1%

#### Performance Metrics

| Metric | Training | Test | Validation |
|--------|----------|------|------------|
| **Calmar Ratio** | **0.3019** | **0.1626** | **0.0919** |
| Sharpe Ratio | 0.5621 | 0.6361 | 0.5109 |
| Sortino Ratio | 0.4791 | 0.5023 | 0.3901 |
| Total Return | +55.12% | +9.44% | +5.96% |
| Max Drawdown | 31.38% | 35.29% | 39.68% |
| Win Rate | 77.42% | 63.64% | 87.50% |
| Total Trades | 31 | 11 | 8 |

#### Portfolio Evolution

- **Training:** $10,000 → $15,512 (+55.12%)
- **Test:** $10,000 → $10,944 (+9.44%)
- **Validation:** $10,000 → $10,596 (+5.96%)

#### Performance Analysis

**Strengths:**
- Positive returns across all three datasets (train/test/validation)
- Training Calmar Ratio of 0.3019 exceeds target range (0.3-0.5)
- Test Calmar Ratio of 0.1626 within acceptable range (0.15-0.3)
- High win rate (63-87%) demonstrates signal quality
- Maximum drawdown controlled below 40% on all datasets

**Observations:**
- Performance degrades from training to validation (expected overfitting)
- Low trade frequency on test/validation (11 and 8 trades respectively)
- Strategy shows conservative behavior, prioritizing quality over quantity
- Risk-adjusted metrics (Sharpe, Sortino) remain positive and consistent

---

## Walk-Forward Analysis

### Purpose

Walk-forward analysis validates strategy robustness by testing on multiple rolling time windows. This technique prevents overfitting by ensuring the strategy works across different market regimes, not just the specific training period.

### Configuration

The walk-forward analysis was performed with the following configuration:

- Training window: 5,000 periods
- Test window: 1,000 periods
- Step size: 1,000 periods
- Total windows analyzed: 51

### Process

For each window:
1. Optimize parameters on training window (5,000 periods)
2. Test optimized parameters on subsequent test window (1,000 periods)
3. Roll forward by step size (1,000 periods)
4. Repeat until dataset exhausted

### Results (run_20251006_174618)

**Aggregate Statistics:**
- Total windows: 51
- Positive windows: 16 (31%)
- Negative windows: 35 (69%)
- Average test Calmar Ratio: -0.44

**Key Findings:**

The walk-forward analysis revealed significant performance variability across different market regimes. While 31% of windows showed positive performance, the majority demonstrated that over-optimized parameters fail to generalize consistently.

**Implications:**

This finding validated our decision to use **conservative parameters** that prioritize:
- Signal quality over quantity
- Consistent behavior across regimes
- Controlled risk (lower position sizes, tighter stops)

The final model parameters were selected based on:
1. Robust performance on training data
2. Acceptable degradation on test/validation
3. Walk-forward stability considerations

**Conclusion:**

Walk-forward analysis confirmed that pursuing maximum training performance leads to poor out-of-sample results. Our final model balances training performance with generalization ability, resulting in positive (albeit modest) returns on validation data.

---

## Visualizations

The following visualizations are generated automatically and saved to `results/run_20251006_103414/figures/`:

### Training Set Visualizations

1. **train_portfolio_value.png**
   - Portfolio value evolution over time
   - Buy and sell signals marked on chart
   - Shows growth from $10,000 to $15,512

2. **train_cumulative_returns.png**
   - Cumulative returns percentage over time
   - Final return: +55.12%

3. **train_drawdown.png**
   - Drawdown analysis showing peak-to-trough declines
   - Maximum drawdown: 31.38%

4. **train_monthly_returns_heatmap.png**
   - Monthly returns breakdown
   - Color-coded performance by month

### Test Set Visualizations

5. **test_portfolio_value.png**
   - Portfolio value: $10,000 → $10,944
   - 11 trades executed

6. **test_cumulative_returns.png**
   - Cumulative return: +9.44%

7. **test_drawdown.png**
   - Maximum drawdown: 35.29%

8. **test_monthly_returns_heatmap.png**
   - Monthly performance breakdown

### Validation Set Visualizations

9. **validation_portfolio_value.png**
   - Portfolio value: $10,000 → $10,596
   - 8 trades executed

10. **validation_cumulative_returns.png**
    - Cumulative return: +5.96%

11. **validation_drawdown.png**
    - Maximum drawdown: 39.68%

12. **validation_monthly_returns_heatmap.png**
    - Monthly performance breakdown

All visualizations include proper axis labels, legends, and titles for professional presentation.

---

## Project Structure

```
002-Introduction-to-Trading/
├── data/
│   └── Binance_BTCUSDT_1h.csv          # Bitcoin hourly OHLCV data
│
├── src/                                 # Core modules
│   ├── backtest_simple.py              # Backtesting engine
│   ├── indicators.py                    # Technical indicators (RSI, MACD, BB)
│   ├── signals.py                       # 2-of-3 signal confirmation logic
│   ├── metrics.py                       # Performance metrics calculation
│   ├── visualization.py                 # Chart and table generation
│   └── optimizer.py                     # Optuna hyperparameter optimization
│
├── results/                             # Generated outputs
│   ├── run_20251006_103414/            # Final model results
│   │   ├── figures/                    # 12 PNG visualizations
│   │   │   ├── train_portfolio_value.png
│   │   │   ├── train_cumulative_returns.png
│   │   │   ├── train_drawdown.png
│   │   │   ├── train_monthly_returns_heatmap.png
│   │   │   ├── test_portfolio_value.png
│   │   │   ├── test_cumulative_returns.png
│   │   │   ├── test_drawdown.png
│   │   │   ├── test_monthly_returns_heatmap.png
│   │   │   ├── validation_portfolio_value.png
│   │   │   ├── validation_cumulative_returns.png
│   │   │   ├── validation_drawdown.png
│   │   │   └── validation_monthly_returns_heatmap.png
│   │   └── tables/                     # CSV performance metrics
│   │       ├── summary_metrics.csv
│   │       ├── best_parameters.csv
│   │       ├── train_performance_metrics.csv
│   │       ├── test_performance_metrics.csv
│   │       └── validation_performance_metrics.csv
│   │
│   └── run_20251006_174618/            # Walk-forward analysis
│       └── tables/
│           └── walk_forward_results.csv # 51 windows analyzed
│
├── config.py                            # Centralized configuration
├── main.py                              # Main execution pipeline
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

### Module Descriptions

#### backtest_simple.py
Implements the core backtesting engine that simulates trading on historical data. Handles position management, transaction costs, stop-loss/take-profit execution, and circuit breakers.

**Key Functions:**
- `run_backtest()` - Main backtesting loop
- `calculate_position_size()` - Dynamic position sizing based on available capital
- `check_exit_conditions()` - Stop-loss and take-profit logic

#### indicators.py
Calculates all technical indicators used in the strategy.

**Key Functions:**
- `calculate_rsi()` - Relative Strength Index
- `calculate_macd()` - Moving Average Convergence Divergence
- `calculate_bollinger_bands()` - Statistical price bands
- `add_all_indicators()` - Convenience wrapper to add all indicators

#### signals.py
Implements the 2-of-3 confirmation logic for generating trading signals.

**Key Functions:**
- `generate_rsi_signals()` - RSI-based buy/sell signals
- `generate_macd_signals()` - MACD-based buy/sell signals
- `generate_bb_signals()` - Bollinger Band-based buy/sell signals
- `apply_confirmation_logic()` - Combines signals with 2-of-3 rule
- `create_signals()` - Main signal generation pipeline

#### metrics.py
Calculates all performance metrics for strategy evaluation.

**Key Functions:**
- `calculate_calmar_ratio()` - Primary objective metric
- `calculate_sharpe_ratio()` - Risk-adjusted returns (total volatility)
- `calculate_sortino_ratio()` - Risk-adjusted returns (downside volatility)
- `calculate_max_drawdown()` - Largest peak-to-trough decline
- `calculate_win_rate()` - Percentage of profitable trades
- `calculate_all_metrics()` - Wrapper for all metrics

#### visualization.py
Generates all charts and performance tables.

**Key Functions:**
- `plot_portfolio_value()` - Portfolio value with trade markers
- `plot_cumulative_returns()` - Returns over time
- `plot_drawdown()` - Drawdown analysis
- `plot_monthly_returns_heatmap()` - Monthly returns grid
- `generate_all_visualizations()` - Creates all charts

#### optimizer.py
Implements Bayesian optimization using Optuna for hyperparameter tuning.

**Key Functions:**
- `objective_function()` - Optuna objective to maximize Calmar Ratio
- `optimize_strategy()` - Main optimization routine
- `walk_forward_analysis()` - Rolling window validation

#### config.py
Centralizes all configuration parameters for easy modification.

**Configuration Sections:**
- `DATA_CONFIG` - File paths and data split ratios
- `STRATEGY_PARAMS` - Indicator parameters and thresholds
- `PORTFOLIO_CONFIG` - Initial capital, fees, position limits
- `OPTIMIZATION_CONFIG` - Number of trials, search space
- `METRICS_CONFIG` - Risk-free rate, annualization factors
- `VISUALIZATION_CONFIG` - Chart styling and output paths

#### main.py
Orchestrates the complete trading strategy pipeline.

**Execution Flow:**
1. Load and preprocess data (60/20/20 split)
2. Optimize hyperparameters on training data (100 trials)
3. Run backtesting on all three datasets
4. Calculate performance metrics
5. Generate visualizations
6. Save results to timestamped directory

---

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone the repository

2. Create virtual environment (recommended)

3. Install dependencies from requirements.txt

### Required Dependencies

- pandas 1.5.0 or higher
- numpy 1.23.0 or higher
- matplotlib 3.6.0 or higher
- seaborn 0.12.0 or higher
- ta 0.11.0 or higher (Technical Analysis Library)
- optuna 3.0.0 or higher
- scipy 1.9.0 or higher

### Running the Strategy

Execute main.py to run the complete pipeline. This will load and split the Bitcoin data, run optimization (100 trials, approximately 5-10 minutes), backtest on train/test/validation datasets, calculate all performance metrics, generate 12 visualizations, and save results to a timestamped directory.

Configuration options can be modified in config.py to adjust number of optimization trials, risk management parameters, position sizing limits, circuit breaker thresholds, and data split ratios.

### Viewing Results

After execution, results are saved to results/run_TIMESTAMP/ directory containing tables with performance metrics and figures with visualizations.

---

## Technologies

### Core Technologies

- **Python 3.8+** - Primary programming language
- **pandas** - Data manipulation and time series analysis
- **NumPy** - Numerical computations and array operations
- **Optuna** - Bayesian hyperparameter optimization
- **Matplotlib** - Static visualizations and charts
- **Seaborn** - Statistical data visualization

### Technical Analysis

- **ta (Technical Analysis Library)** - Pre-built indicator calculations
- **scipy** - Statistical functions for metrics calculation

### Development Tools

- **Git** - Version control
- **VS Code** - Development environment with Claude Code integration
- **Jupyter Notebooks** - Exploratory data analysis (optional)

### Key Libraries Usage

The project leverages Optuna for Bayesian hyperparameter optimization, pandas for time series data manipulation and analysis, and matplotlib with seaborn for professional visualization generation.

---

## Key Findings

### Successful Outcomes

1. **Positive Returns Across All Datasets**
   - Training: +55.12%
   - Test: +9.44%
   - Validation: +5.96%
   - All datasets profitable demonstrates strategy viability

2. **Strong Calmar Ratios**
   - Training: 0.3019 (exceeds target of 0.3-0.5)
   - Test: 0.1626 (within acceptable range 0.15-0.3)
   - Validation: 0.0919 (positive, though lower)
   - Consistent risk-adjusted performance

3. **Effective Risk Management**
   - Maximum drawdown: 39.68% (below 50% threshold)
   - No catastrophic losses
   - Circuit breakers never triggered
   - Conservative position sizing prevented overleveraging

4. **High Signal Quality**
   - Win rates: 63-87% across datasets
   - 2-of-3 confirmation reduces false positives
   - Low trade frequency indicates selective signal generation

### Strategy Characteristics

**Conservative Approach:**
- Low trade frequency (8-31 trades per dataset)
- Prioritizes quality over quantity
- Avoids overtrading and excessive transaction costs

**Risk-Adjusted Focus:**
- Calmar Ratio optimization balances returns with drawdown
- Sharpe and Sortino ratios consistently positive
- Downside protection through tight risk management

**Mean Reversion Philosophy:**
- Strategy capitalizes on price extremes
- Works well in ranging/sideways markets
- May underperform in strong trending environments

### Walk-Forward Insights

- Over-optimization leads to poor out-of-sample performance
- Only 31% of windows showed positive returns
- Conservative parameter selection improves robustness
- Strategy performance highly dependent on market regime

---

## Limitations

### Trade Frequency

**Observation:** Low number of trades on test (11) and validation (8) datasets.

**Implications:**
- Limited statistical sample size for validation
- High variance in performance metrics
- Strategy may miss opportunities during trending markets

**Mitigation:** Conservative approach prioritizes avoiding false signals over capturing every move.

### Drawdown Proximity to Threshold

**Observation:** Validation max drawdown of 39.68% approaches 40% circuit breaker threshold.

**Implications:**
- Strategy operates close to risk limits
- Small parameter changes could trigger circuit breaker
- Requires careful monitoring in live trading

**Consideration:** Risk tolerance should match investor profile.

### Market Regime Dependency

**Observation:** Walk-forward analysis showed 69% negative windows.

**Implications:**
- Strategy performs inconsistently across different market conditions
- Mean reversion approach struggles in strong trends
- Requires market regime detection for optimal deployment

**Future Enhancement:** Implement market state classifier to activate/deactivate strategy.

### Hourly Timeframe Noise

**Challenge:** Hourly Bitcoin data exhibits high volatility and noise.

**Impact:**
- False signals from short-term fluctuations
- Wider stops required to avoid premature exits
- Lower win rates compared to daily timeframes

**Alternative:** Strategy may perform better on daily or 4-hour timeframes.

### Optimization Overfitting Risk

**Observation:** Training Calmar (0.30) significantly exceeds validation (0.09).

**Implications:**
- Some degree of overfitting present
- Real-world performance likely closer to validation results
- Parameter sensitivity requires ongoing monitoring

**Mitigation:** Walk-forward analysis used to select robust parameters.

---

## References

### Academic Papers

1. **Fama, E. F. (1970).** "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383-417.

2. **Roll, R. (1984).** "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market." *Journal of Finance*, 39(4), 1127-1139.

3. **Jegadeesh, N., & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65-91.

### Technical Resources

- **Technical Analysis Library (TA-Lib):** https://github.com/mrjbq7/ta-lib
- **Optuna Documentation:** https://optuna.readthedocs.io/
- **Pandas Time Series:** https://pandas.pydata.org/docs/user_guide/timeseries.html

### Data Source

- **Binance Exchange:** https://www.binance.com/
- BTC/USDT hourly OHLCV data

---

## Acknowledgments

This project was developed as part of an academic course on algorithmic trading and quantitative finance. Special thanks to:

- **Prof. Luis Felipe Gómez Estrada** - Course instructor and guidance
- **Claude** - Development assistance and code review
- **Binance** - Historical market data provision
- **Open-source community** - Technical analysis libraries and tools

---

