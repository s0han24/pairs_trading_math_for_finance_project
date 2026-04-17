# Pairs Trading Strategies

This repository currently implements a cointegration-based pairs trading pipeline on the NIFTY100 universe, with shared backtesting utilities reused across strategy variants.

## Current Scope

Implemented:
1. Cointegration-based pair selection (Engle-Granger and Johansen)
2. Correlation pre-filtering (Pearson)
3. Spread modeling and z-score-based signal generation
4. Shared walk-forward backtesting with portfolio metrics

Planned in the project outline but not implemented in this codebase yet:
1. Negative-correlation-based strategy variants
2. Baseline portfolios (40-60, equal-weight, greedy, market-only)

## Universe

The stock universe is NIFTY100. Tickers are read from `ind_nifty100list.csv` in `cointegration_based/config/universe.py`, and `.NS` suffixes are added where needed.

## Project Structure

```text
pairs_trading_math_for_finance_project/
├── cointegration_based_tests.py
├── ind_nifty100list.csv
├── correlation_statistics/
│   └── statistics.py
├── backtests/
│   ├── metrics.py
│   └── walk_forward_pipeline.py
└── cointegration_based/
    ├── backtest/
    │   ├── backtesting.py
    ├── config/
    │   └── universe.py
    ├── data/
    │   └── downloader.py
    ├── pairs/
    │   ├── cointegration.py
    │   └── filtering.py
    ├── spread_models/
    │   ├── hedge_ratio.py
    │   ├── spread.py
    │   └── zscore.py
    └── strategy/
        ├── pipeline.py
        └── signals.py
```

## Workflow

1. Universe load from `ind_nifty100list.csv`
2. Price download from Yahoo Finance
3. Correlation filtering to keep top candidate pairs
4. Cointegration testing to keep tradable pairs
5. Spread and z-score computation (`simple` or `ou`)
6. Signal generation (long/short spread with mean-reversion exits)
7. Shared walk-forward backtest and metrics calculation

## How to Run

Run the main experiment script:

```bash
python cointegration_based_tests.py
```

The script builds a `CointegrationPipeline`, passes it into the shared walk-forward backtester, and compares multiple strategy variants, including:
1. Engle-Granger vs Johansen
2. Sorted-by-correlation vs p-value-first selection
3. `simple` vs `ou` z-score method

It prints a metrics table (final capital, max drawdown, Sharpe, alpha, beta, annual volatility) and plots equity curves.

## Dependencies

Install required packages:

```bash
pip install yfinance pandas numpy statsmodels matplotlib scikit-learn prettytable
```

## Future Work
- Implement more correlation statistics
- ML or DL based approaches
- Baselines
- Extend the cointegration approach to include stochastic modeling based approaches such as:
    1. Time-varying OU
    2. Kalman Filter
