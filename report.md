# Pairs Trading on the NIFTY100 Universe
### Math for Finance — Project Report

## Table of Contents
1. [Motivation & Overview](#1-motivation--overview)
2. [Methodology](#2-methodology)
   - 2.1 [Pair Selection (In-Sample)](#21-pair-selection-in-sample)
   - 2.2 [Spread & Z-Score Models](#22-spread--z-score-models)
   - 2.3 [Signal Generation](#23-signal-generation)
   - 2.4 [Prediction-Based Strategies](#24-prediction-based-strategies)
   - 2.5 [Backtesting Framework](#25-backtesting-framework)
   - 2.6 [Baselines](#26-baselines)
3. [Results & Analysis](#3-results--analysis)

## 1. Motivation & Overview

Statistical arbitrage — and pairs trading in particular — is one of the most well-studied market-neutral strategies in quantitative finance. The core idea is that two historically co-moving assets will occasionally diverge from their long-run equilibrium. By simultaneously going long on the underperformer and short on the overperformer, a trader can profit from the eventual reversion to this equilibrium, all without taking a directional bet on the overall market.

Pairs trading is attractive for several reasons:

- **Market neutrality**: The simultaneous long-short structure makes the strategy largely insensitive to broad market movements (low or negative beta).
- **Statistical grounding**: Entry and exit signals are derived from statistically rigorous tests (cointegration, z-score thresholds) rather than heuristics.
- **Risk control**: The structure naturally limits exposure, and stop-loss rules cap downside in diverging regimes.

This project implements a systematic pairs trading framework targeting the **NIFTY100 universe** — the 100 largest companies listed on the National Stock Exchange of India. The evaluation period covers approximately 2014–2026, encompassing multiple distinct market regimes including pre-COVID growth, the 2020 market crash and recovery, and subsequent volatile periods.

The project explores two complementary strategies:
1. **Cointegration-based pairs trading**: Classical statistical spread trading using Engle-Granger and Johansen cointegration tests with two z-score parameterizations.
2. **ML-based cross-sectional prediction**: A replication of the Krauss, Do & Huck (2017) framework using ensemble classifiers to predict relative outperformance.

## 2. Methodology

### 2.1 Pair Selection (In-Sample)

Selecting tradeable pairs from a universe of $N \approx 100$ stocks involves evaluating $O(N^2)$ candidate pairs. Three broad families of pair selection exist:

- **Distance-based**: Euclidean distance between normalized (z-scored) price series.
- **Correlation-based**: Pearson correlation of log returns.
- **Cointegration-based**: Formal statistical tests (Engle-Granger or Johansen) that guarantee mean-reversion of the spread.

While cointegration-based methods are the most theoretically robust — correlation is a *necessary* but not *sufficient* condition for a stationary spread — they are computationally intensive for large universes.

#### Two-Stage Filtering

To balance statistical rigor with computational efficiency, we adopt a **two-stage approach**:

**Stage 1 — Correlation Screening** (`cointegration_based/pairs/filtering.py`):
- Compute the Pearson correlation matrix of log returns across all NIFTY100 assets.
- Retain the top $n$ pairs (default $n = 20$) with correlation above a threshold $\rho_{\min} = 0.80$.
- This drastically reduces the candidate set before the expensive cointegration step.

**Stage 2 — Cointegration Testing** (`cointegration_based/pairs/cointegration.py`):
- Apply either the **Engle-Granger** two-step test (OLS regression + Dickey-Fuller test on residuals, via `statsmodels`) or the **Johansen** vector error-correction test (`det_order=0, k_ar_diff=1`) to the screened pairs.
- Retain pairs with generalized p-value $< P_{\text{threshold}} = 0.05$.
- Rank surviving pairs by their **in-sample Sharpe ratio** and select the top $k$ pairs (default $k = 5$) for trading.

The key insight driving this design is that **cointegration guarantees a stationary spread**, which is the actual trading signal. Correlation alone does not.

### 2.2 Spread & Z-Score Models

#### Spread Construction

For each selected pair $(y, x)$, the spread is defined as:

$$\text{Spread}_t = y_t - \beta x_t$$

where $\beta$ is the **hedge ratio**, estimated via OLS over the full training window (static $\beta$). Dynamic alternatives (rolling OLS, Kalman filter) are noted as future extensions.

#### Z-Score Parameterizations

Two methods are implemented for converting the raw spread into a standardized signal:

**Method 1 — Simple Rolling Window** (`method="simple"`):

$$Z_t = \frac{\text{Spread}_t - \text{SMA}_{60}(\text{Spread})}{\text{StdDev}_{60}(\text{Spread})}$$

Uses a standard 60-day rolling mean and standard deviation.

**Method 2 — Ornstein-Uhlenbeck (OU) Process** (`method="ou"`):

The spread is modeled as a continuous-time mean-reverting process:

$$dS_t = \theta(\mu - S_t)\,dt + \sigma\,dW_t$$

Parameters are estimated via a discrete-time linear regression of $\Delta S_t = a + b S_{t-1}$, giving:

$$\hat{\theta} = -b, \qquad \hat{\mu} = \frac{a}{1 - \hat{\theta}}$$

The OU-based z-score uses the *expected path* of reversion:

$$Z_t = \frac{\text{Spread}_t - E_{\text{OU}}}{\text{StdDev}_{\text{OU}}}$$

where $E_{\text{OU}} = \mu + (S_t - \mu)e^{-\hat{\theta} \cdot \text{window}}$. This is more statistically grounded than the simple rolling window, as it conditions on the current spread level and the estimated reversion speed.

### 2.3 Signal Generation

Given a z-score time series $\{Z_t\}$, positions are generated according to the following threshold rules.
Note: $\lvert Z_t \rvert$ denotes the absolute value of $Z_t$.

| Condition | Position | Action |
| --- | --- | --- |
| $Z_t > +2.0$ | -1 (Short spread) | Short $y$, Long $x$ |
| $Z_t < -2.0$ | +1 (Long spread) | Long $y$, Short $x$ |
| $\lvert Z_t \rvert \leq 0.5$ | 0 (Flat) | Close position |
| Stop-loss triggered | 0 (Flat) | Emergency exit |
| Otherwise | Unchanged | Hold |

The **stop-loss** exits any open position if the spread diverges beyond a configurable threshold (default: $\lvert Z_t \rvert > 5.0$), capping the tail risk during regime shifts.

**Portfolio aggregation**: Capital is allocated equally across all $k$ active pairs. Total portfolio wealth carries forward continuously across walk-forward windows.

### 2.4 Prediction-Based Strategies

In addition to the cointegration framework, we implement a cross-sectional ML approach following **Krauss, Do & Huck (2017)**.

#### Task Formulation

Rather than modelling individual price series, the classifier is trained to predict **cross-sectional relative performance**: for each stock $s$ on each day $t$, the binary label is $y_{s,t} = \mathbf{1}[\text{next-day return of } s > \text{cross-sectional median}]$.

At test time, stocks are ranked by $P(\text{outperform})$ and the strategy goes **long on the top-k** and **short on the bottom-k** stocks, forming a dollar-neutral portfolio.

#### Feature Engineering

Passing the full year of daily returns (252 features) would be too high-dimensional for the available training samples. Instead, we construct **31 lagged-return features** per stock per day:
- **Short-range**: Daily returns for lags $m \in \{1, 2, \ldots, 20\}$ (20 features).
- **Long-range**: Returns over 20-day intervals for lags $m \in \{40, 60, 80, \ldots, 240\}$ (11 features).

*Note*: We also attempted to use time-series models (ARIMA, MomentFM) for this task, but they significantly underperformed, as they are not designed for cross-sectional ranking tasks.

#### Models

Four base classifiers are trained:

| Model | Architecture / Configuration |
| --- | --- |
| DNN (MLP) | Layers: 31-31-10-5-2; ReLU; Adam; alpha = 1e-5; 400 epochs |
| GBT | 100 trees, max depth 3, learning rate 0.1, 15 features per split |
| Random Forest (RAF) | 1000 trees, max depth 20, sqrt(31) ≈ 6 features per split |
| XGBoost (XGB) | 100 trees, max depth 3, LR 0.1; hyperparameters tuned via Optuna |

An **equal-weight ensemble** averages the outperformance probabilities across all enabled models: $P_{\text{ENS}} = \frac{1}{M}\sum_{m} P_m$.

A **confidence gate** prevents trading on uncertain days: if the mean probability of the top-k stocks is below a threshold (default: 0.55), the portfolio holds cash for that day.

#### XGBoost Hyperparameter Tuning

XGBoost was additionally tuned using **Optuna** (Bayesian optimization) across all walk-forward folds, searching over:
`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`.

### 2.5 Backtesting Framework

All strategies are evaluated using **walk-forward (out-of-sample) backtesting** to prevent look-ahead bias. The framework rolls a training window forward over time:

```
Train [3 yr] → Test [1 yr]
Train [3 yr] → Test [1 yr]
...
```

Covering approximately 2014–2026. Portfolio wealth carries forward continuously across windows.

**Performance metrics** computed across all out-of-sample periods:

| Metric | Description |
| --- | --- |
| Final Capital | Terminal portfolio value (starting capital = 100 USD) |
| SR (Sharpe Ratio) | Annualised excess return / annualised volatility (252 days, 0% risk-free) |
| MD (Max Drawdown) | Largest peak-to-trough decline in portfolio value |
| Alpha | Intercept from regressing strategy returns on NSEI daily returns |
| Beta | Slope from regressing strategy returns on NSEI daily returns |
| Sigma | Annualised standard deviation of daily returns (%) |

### 2.6 Baselines

Five passive/heuristic strategies provide performance benchmarks. All baselines use the same walk-forward windows and metrics interface.

| Strategy | Description |
| --- | --- |
| 40-60 | 40% in cash at 6.5% p.a. (Indian T-bill) + 60% tracking NSEI index |
| EqualWeight | Daily rebalance to equal weight across all NIFTY100 stocks |
| EW_BuyHold | Equal weight on day 1, no rebalancing thereafter |
| Greedy | 100% in the stock with the highest cumulative gain in the training window |
| Market | 100% in NSEI (NIFTY 50 index) |

Shared assumptions: starting capital 100 USD, risk-free rate 6.5% p.a. (daily: $(1.065)^{1/252} - 1$), benchmark: NSEI.

## 3. Results & Analysis

All strategies start from capital = 100 (USD). MD = Max Drawdown, SR = Sharpe Ratio (annualised, 0% risk-free), alpha and beta are from regressing strategy returns on NSEI daily returns, sigma = annualised volatility (%).

### 3.1 Cointegration-Based Strategies (k = 5)

Results with and without stop-losses, using rolling 60-day z-scores (Pearson) vs OU z-scores (staticOU). The stop-loss triggers at 3 standard deviations of the spread.

| Strategy | Final Capital | MD | SR | Alpha | Beta | Sigma |
| --- | --- | --- | --- | --- | --- | --- |
| EG-Pearson-staticOU-StopLoss | 315.47 | -26.72 | 0.65 | 0.1858 | -0.1059 | 24.74 |
| EG-Pearson | 307.17 | -36.22 | 0.86 | 0.1522 | -0.0479 | 16.45 |
| EG-Pearson-StopLoss | 307.17 | -36.22 | 0.86 | 0.1522 | -0.0479 | 16.45 |
| Johansen-Pearson | 204.66 | -66.45 | 0.43 | 0.1273 | -0.0397 | 27.17 |
| Johansen-Pearson-StopLoss | 204.66 | -66.45 | 0.43 | 0.1273 | -0.0397 | 27.17 |
| EG-Pearson-staticOU | 167.24 | -49.71 | 0.35 | 0.2465 | -0.4320 | 41.16 |
| Johansen-Pearson-staticOU-StopLoss | 161.83 | -32.39 | 0.35 | 0.0767 | 0.0289 | 24.08 |
| Johansen-Pearson-staticOU | 133.08 | -40.59 | 0.26 | 0.0776 | -0.0139 | 28.99 |

- **Engle-Granger + simple z-score** is the strongest variant (SR = 0.86, 3.1x return) with near-zero market exposure (beta ≈ -0.048).
- **Stop-loss has no effect** on the simple z-score — positions revert naturally before the threshold is hit. For the OU z-score, the stop-loss cuts max drawdown from -49.71 to -26.72 at the cost of a lower SR.
- **Johansen consistently underperforms** EG, likely because the VECM structure is less suited to short training windows.
- All variants show **negative beta**, confirming genuine market neutrality.

### 3.2 ML-Based Strategies (k = 5)

#### Untuned Models

The Gate column is the minimum mean probability of the top-k stocks required to trade on a given day (0.0 = no gate, trades every day).

| Strategy | Gate | Final Capital | MD | SR | Alpha | Beta | Sigma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ML-XGB-k5 | 0.0 | 473.00 | -11.54 | 1.61 | 0.1735 | 0.0386 | 11.38 |
| ML-XGB-k5 | 0.5 | 399.90 | -12.71 | 1.43 | 0.1578 | 0.0253 | 11.48 |
| ML-RAF-k5 | 0.0 | 284.10 | -18.99 | 1.16 | 0.1197 | 0.0188 | 10.70 |
| ML-Ensemble-k5 | 0.5 | 291.74 | -25.52 | 1.16 | 0.1182 | 0.0391 | 11.01 |
| ML-Ensemble-k5 | 0.0 | 256.90 | -30.45 | 1.02 | 0.1067 | 0.0270 | 11.03 |
| ML-RAF-k5 | 0.5 | 243.29 | -18.83 | 0.99 | 0.1073 | -0.0025 | 10.82 |
| ML-DNN-k5 | 0.0 | 188.54 | -33.31 | 0.73 | 0.0796 | -0.0086 | 10.64 |
| ML-DNN-k5 | 0.5 | 164.23 | -16.14 | 0.58 | 0.0605 | 0.0059 | 10.68 |

#### Tuned XGBoost

XGBoost hyperparameters were tuned with Optuna (Bayesian optimisation) across all walk-forward folds, searching over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`.

| Strategy | Final Capital | MD | SR | Alpha | Beta | Sigma |
| --- | --- | --- | --- | --- | --- | --- |
| ML-XGB-k5 (tuned) | 449.83 | -7.98 | 2.21 | 0.1720 | 0.0067 | 7.83 |

- **Tuned XGBoost achieves the best risk-adjusted performance overall** (SR = 2.21, MD = -7.98%), cutting volatility nearly in half vs. the untuned version (7.83% vs. 11.38%).
- **Untuned XGB (no gate)** has the highest raw final capital (473.00) but a much higher drawdown (-11.54%) and volatility.
- The **confidence gate** is most beneficial for noisy models: DNN drawdown improves from -33% to -16% with gate = 0.5. For XGB, it reduces final capital without a commensurate drawdown improvement.
- **Ensemble underperforms XGB alone** — averaging with weaker models dilutes XGB's signal.
- All ML strategies show **near-zero or slightly positive beta**, in contrast to the negative-beta cointegration strategies.

### 3.3 Baselines

All baselines share the same walk-forward windows and starting capital. ARIMA models were also tested in the ML cross-sectional framework but failed to produce positive alpha.

| Strategy | Final Capital | MD | SR | Alpha | Beta | Sigma |
| --- | --- | --- | --- | --- | --- | --- |
| Greedy | 128580.36 | -52.93 | 1.41 | 0.7207 | 1.1792 | 62.49 |
| EW_BuyHold | 1633.68 | -36.52 | 1.41 | 0.1744 | 0.9764 | 21.81 |
| EqualWeight | 1534.34 | -37.37 | 1.39 | 0.1656 | 0.9923 | 21.63 |
| Market | 440.72 | -38.44 | 0.97 | 0.0290 | 1.0024 | 17.15 |
| 40-60 | 344.56 | -22.71 | 1.26 | 0.0539 | 0.5789 | 10.57 |
| ARIMA-ARIMA202-k5 | 114.43 | -25.87 | 0.19 | 0.0202 | 0.0059 | 11.21 |
| ARIMA-AR1-k5 | 104.68 | -32.30 | 0.10 | 0.0072 | 0.0173 | 11.14 |
| ARIMA-AR5-k5 | 103.51 | -22.06 | 0.09 | 0.0106 | -0.0011 | 11.37 |
| ARIMA-ARIMA101-k5 | 75.55 | -36.20 | -0.23 | -0.0286 | 0.0129 | 11.18 |

- **Greedy** produces extreme nominal gains but is not investable (beta = 1.18, MD = -53%).
- **EW_BuyHold / EqualWeight** are strong passive benchmarks (SR ≈ 1.41) but carry full market risk (beta ≈ 1.0).
- **Market** (440.72, SR = 0.97) is the most natural comparison point — tuned XGB and untuned XGB (473) both beat it on all dimensions.
- **ARIMA models** barely preserve capital and yield near-zero alpha, confirming that univariate time-series forecasts are ill-suited to the cross-sectional ranking task.

### 3.4 Discussion

1. **Z-score choice dominates test selection**: Simple vs. OU z-score has a larger impact than EG vs. Johansen. Simple z-score wins on SR; OU z-score reduces drawdown.
2. **Stop-losses are conditionally useful**: Beneficial for OU spread strategies during regime shifts, negligible for simple z-score (positions revert before the threshold).
3. **ML benefits from cross-sectional framing**: Predicting relative outperformance rather than absolute returns unlocks cross-sectional signals unavailable to cointegration methods.
4. **Tuning matters for XGBoost**: Optuna tuning raises SR from 1.61 to 2.21 and halves volatility (11.4% to 7.8%), at a slight cost in final capital.
5. **XGBoost outperforms the original DNN**: Unlike Krauss et al. (2017) on S&P 500, the NIFTY100 return structure is better captured by gradient-boosted trees than by the MLP architecture.

## Future Work

- **Dynamic hedge ratios**: Kalman filter or rolling OLS for time-varying $\beta$ estimation.
- **Time-varying OU parameters**: Fit OU parameters in a rolling fashion to adapt to changing spread dynamics.
- **Deep Reinforcement Learning**: Learn position-sizing and entry/exit rules end-to-end.
- **Extended correlation statistics**: Spearman rank correlation, information coefficient, lead-lag analysis.
- **Risk measures**: VaR, CVaR in addition to Sharpe and max drawdown.
- **Larger k**: Test with more pairs to assess whether the ML advantage persists at k = 10.
