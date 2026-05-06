# Pairs Trading Framework: Math for Finance

This repository implements a systematic statistical arbitrage (pairs trading) pipeline deployed on the **NIFTY100 universe**. It focuses heavily on statistical cointegration, rolling window backtests, and alternative z-score construction paradigms (simple moving average vs. Ornstein-Uhlenbeck processes).

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
- **Selection**: We take the top $k$ pairs (default `5` or `10`), sorted by **In-Sample Sharpe Ratio** to prioritize historically profitable cointegrated pairs (alternatively, pairs can be sorted by the lowest p-value or highest correlation).

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
- **Stop Loss**: Liquidate position (return to neutral) if the spread diverges beyond an unacceptable threshold.
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

---

## 6. Baselines (`baselines.py`)

Five simple baseline strategies are implemented to benchmark the cointegration pairs-trading approach. All baselines share the same walk-forward windows (2-year train → 2-year test) and the same metrics interface.

### Strategy Summary

| Strategy | Description |
|---|---|
| **40-60** | Hold 40% in cash (earning 6.5% risk-free) and 60% tracking the ^NSEI index. |
| **EqualWeight** | Split capital equally across all stocks and rebalance back to equal weights every day. |
| **EW_BuyHold** | Split capital equally across all stocks on day 1, then do nothing and let winners drift. |
| **Greedy** | Each training window, find the stock with the highest 2-year cumulative gain; bet 100% on it for the next 2 years. |
| **Market** | Put 100% in the ^NSEI index — the plain "just buy the index" benchmark. |

### Assumptions 

| Assumption | Value | Notes |
|---|---|---|
| Starting capital | **$100** | Initial portfolio value at the start of the first window |
| Risk-free rate | **6.5% p.a.** | Indian T-bill proxy; used only by the 40-60 baseline; converted to daily as $(1.065)^{1/252} - 1$ |
| Benchmark | **^NSEI (NIFTY 50)** | All alpha/beta computed against the NIFTY 50 index daily returns |

## 7. Prediction based Approaches

This section details the machine learning and time-series forecasting pipelines used for statistical arbitrage.

### 7.1 ML-Based Pipeline (Krauss et al. 2016)
Located in `prediction_based/ml_based/pipeline.py`, this approach is a faithful implementation of the strategy proposed by Krauss, Do & Huck (2017) [1].

#### Feature Engineering & Labeling
- **Features**: 31 lagged return features $R(m)$ for $m \in \{1, \dots, 20, 40, 60, \dots, 240\}$.
- **Labeling**: Binary classification. A stock is labeled $1$ if its next-day return outperforms the cross-sectional median of all stocks in the universe, and $0$ otherwise.
- **Scaling**: Features are standardized using a `StandardScaler` fitted on the training window.

#### Model Architectures
The framework ensembles four types of classifiers:
1. **Deep Neural Network (DNN)**: A multi-layer perceptron with architecture 31-31-10-5 (as per paper topology).
2. **Gradient Boosted Trees (GBT)**: 100 trees with depth 3.
3. **Random Forest (RAF)**: 1000 trees with depth 20.
4. **XGBoost (XGB)**: Added as an alternative to GBT with similar hyperparameters. Hyperparameter tuning using Optuna was performed for XGBoost to achieve a sharpe of ~2.2! Which really shows how well these approaches work.

Note that transaction costs are ignored for metric calculations throughout the project. 

#### Trading Logic & Confidence Gating
- **Ensembling**: Predictions are generated as the equal-weighted average of the probabilities $P(outperform)$ from all fitted models.
- **Ranking**: Stocks are ranked by their ensembled probability. The strategy goes **Long top-k** and **Short bottom-k** (dollar-neutral).
- **Confidence Gate (Middle-Censoring)**: To avoid trading on low-conviction signals, the strategy only enters positions if the average probability of the top-k stocks exceeds a threshold (e.g., `0.55`). This effectively "censors" the uncertain middle of the ranking. Set this to zero to disable the gating mechanism. 

---

### 7.2 ARIMA Baseline
Located in `prediction_based/arima/pipeline.py`, this acts as a naive time-series baseline to benchmark the ML approaches.

- **Model**: A rolling ARIMA(2, 0, 2) model is fitted per-stock on log-returns using a 252-day lookback window.
- **Forecast**: Generates 1-step-ahead point forecasts for next-day log-returns.
- **Execution**: Ranks stocks by forecast returns; long top-k, short bottom-k.
- **Gating**: Only trades if the mean forecast return of the top-k stocks is positive (`min_ret_threshold > 0`).

---

### 7.3 Shared "Warmup" Logic
Both prediction pipelines implement a **Warmup Prefix** mechanism. To ensure the strategy can trade from day 1 of the test window, the last 240 days of the training window are prepended to the test data. This prevents a "cold-start" period where the models would otherwise wait for enough history to generate the first set of features.

---

### 7.4 MOMENT-based ML Pipeline (Future Work)

We also implemented a pipeline in `prediction_based/ml_based/moment_pipeline.py` that integrates the **MOMENT (Foundation Model)** with the ML strategy. However, this implementation is currently **non-working** and serves as a placeholder for future research.

#### Architecture
This pipeline follows the same ensemble structure as the standard ML approach but replaces the custom-trained models with a fine-tuned MOMENT foundation model. The architecture is defined in `prediction_based/ml_based/model_configs.py`.

#### Current Status
**DO NOT USE**. This is still under development and currently non-working. It serves as a placeholder for future work.


## References

\[1\] Christopher Krauss, Xuan Anh Do, Nicolas Huck, Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500, European Journal of Operational Research, 2017
\[2\] MOMENT: A Family of Open Time-series Foundation Models
\[3\] Caldeira, João and Moura, Guilherme Valle, Selection of a Portfolio of Pairs Based on Cointegration: A Statistical Arbitrage Strategy (January 4, 2013).
