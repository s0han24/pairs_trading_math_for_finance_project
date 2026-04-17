# Pairs Trading Framework: Math for Finance

This repository implements a systematic statistical arbitrage (pairs trading) pipeline deployed on the **NIFTY100 universe**. It focuses heavily on statistical cointegration, rolling window backtests, and alternative z-score construction paradigms (simple moving average vs. Ornstein-Uhlenbeck processes).

**Goal of this README**: Provide absolute clarity regarding the architecture, math, data structures, and assumptions embedded into the python files so that an LLM or developer can infer the system's exact state without opening the source code. Note: There is an `ml_based` folder not explicitly detailed in this document as it pertains to an isolated random-forest/deep-learning experiment.

For now, ignore the `ml_based/` folder and `ml_based_tests.py` which are still being worked on.

---

## 1. Project Overview & Architecture

The framework is structured into shared modules and strategy-specific modules:
1. **Universe Setup**: (`config/universe.py`) Contains the NIFTY100 tickers loaded from a CSV.
2. **Data Download**: (`data/downloader.py`) Downloads adjusted close prices using `yfinance`.
3. **Correlation Stats**: (`correlation_statistics/statistics.py`) Utilities for correlation computation across assets.
4. **Strategy Pipeline**: (`cointegration_based/strategy/pipeline.py`) The main `CointegrationPipeline` class encapsulating the `fit()` and `backtest()` behavior.
5. **Walk-Forward Evaluation**: (`backtests/walk_forward_pipeline.py` & `backtests/metrics.py`) Reusable 2-year train, 2-year test rolling boundaries and performance tracking.

---

## 2. In-Sample Training (The `fit` Phase)

The initial problem of pairs trading requires narrowing down $O(N^2)$ pairs into a tradable subset $k$. The framework uses a two-stage filter:

### Stage 1: Correlation Filtering
Located in `cointegration_based/pairs/filtering.py`.
- **Input**: A price correlation matrix (Pearson by default).
- **Action**: Sorts all pairs by correlation score and takes the top $n$ pairs (parameter `n` defaults to `20` or `50`).
- **Threshold**: Requires correlations greater than `corr_threshold` (default `0.8`).

### Stage 2: Cointegration Testing
Located in `cointegration_based/pairs/cointegration.py`.
We test the top $n$ correlated pairs using either:
- **Engle-Granger**: Two-step OLS with Dickey-Fuller on residuals (using `statsmodels.tsa.stattools.coint`).
- **Johansen**: Vector Error Correction Model (using `statsmodels.tsa.vector_ar.vecm.coint_johansen` with `det_order=0, k_ar_diff=1`).
- **P-Value assignment**: Pairs scoring a generalized p-value $< P_{threshold}$ (default `0.05`) are retained.
- **Selection**: We take the top $k$ pairs (default `5` or `10`), either sorted by the lowest p-value (`sort_by_corr=False`) OR sorted by the highest correlation (`sort_by_corr=True`).

---

## 3. Out-Of-Sample Trading (The `backtest` Phase)

Once $k$ pairs are locked in from the training window, we execute them out-of-sample over the subsequent 2 years through the `backtest_pairs` function (`cointegration_based/backtest/backtesting.py`).

### Spread Construction
For each chosen pair $(y, x)$, computed inside `cointegration_based/spread_models/spread.py`:
1. Use `estimate_hedge_ratio` (`cointegration_based/spread_models/hedge_ratio.py`) to deduce dynamic or static $\beta$.
2. The spread series is computed strictly as: $Spread_t = y_t - \beta x_t$.

### Z-Score Parameterizations (`cointegration_based/spread_models/zscore.py`)
To isolate mean-reverting deviations, we compute $Z_t$. Two methods are supported:
1. **Simple (`method="simple"`)**:
   Standard 60-day rolling window. 
   $$Z_t = \frac{Spread_t - SMA_{60}(Spread)}{StdDev_{60}(Spread)}$$
2. **Ornstein-Uhlenbeck (`method="ou"`)**:
   Models the spread as $d S_t = \theta (\mu - S_t) dt + \sigma d W_t$.
   - Using linear regression: $\Delta S_t = a + b S_{t-1}$ to estimate $\theta = -b$ and asymptotic mean $\mu = \frac{a}{1 - \theta}$.
   - Expected mean: $E_{OU} = \mu + (S_t - \mu)e^{-\theta \times window}$.
   - Expected standard deviation derived similarly. 
   - $$Z_t = \frac{Spread_t - E_{OU}}{StdDev_{OU}}$$.

### Signal Generation (`cointegration_based/strategy/signals.py`)
For a given Z-score timeseries:
- **Entry**: When $Z_t > 2.0$, Position = $-1$ (Short spread: Short $y$, Long $x$). When $Z_t < -2.0$, Position = $+1$ (Long spread: Long $y$, Short $x$).
- **Exit**: When $|Z_t| \le 0.5$, Position = $0$ (Flatten).
- **Hold**: Otherwise, previous position is maintained.

### Portfolio Aggregation
Allocates capital equally among the $k$ pairs. Total portfolio wealth carries forward across windows.

---

## 4. Backtest Engine & Metrics

The `run_walk_forward_backtest()` function (`backtests/walk_forward_pipeline.py`) steps through the following overlapping windows:
- 2014-2016 Train $\rightarrow$ 2016-2018 Test
- 2016-2018 Train $\rightarrow$ 2018-2020 Test
... up to 2026.

Performance is collated across out-of-sample stretches evaluating:
- **Max Drawdown**: Highest peak to lowest trough drop (`backtests/metrics.py`).
- **Sharpe Ratio**: Annualized excess returns / annualized volatility (assuming 252 freq, 0.0 risk-free rate).
- **Alpha & Beta**: Computed via covariance against an equally weighted portfolio benchmark of the target universe.

---

## 5. Main Entry points
The primary driver scripts are:
- `cointegration_based_tests.py`: Loops across different methods ('engle-granger', 'johansen'), z-score strategies ('simple', 'ou'), and boolean sorting to evaluate strategy outputs.
- `ml_based_tests.py`: Similar execution but testing the alternate Machine Learning paradigm (`ml_based/strategy/pipeline.py`).

## Future Work
- Implement more correlation statistics
- Sort by in-sample Sharpe Ratio instead of p-values and corr values.
- Add stop-loss and risk measure calculation
- ML or DL based approaches
- Baselines
- Extend the cointegration approach to include stochastic modeling based approaches such as:
    1. Time-varying OU
    2. Kalman Filter
