# Pairs Trading Framework

### Problem Setup:
A pairs trading strategy identifies two co-moving assets that have temporarily diverged in price. The strategy is to go long on the underperformer and short on the overperformer, betting on the mean reverting property of the assets. Another component of the strategy is risk management to cap losses.

We will start off with a mean-variance optimization approach that simply picks the tangency portfolio and try to implement a spread modeling based pairs trading strategy.

### Outline:
1. The Universe
2. Finding inter-asset correlations
3. Build pairs
4. Expected return and volatility prediction
5. Portfolio creation
6. Evaluation metrics/Backtesting
7. Baselines

### The Universe:
Our universe will be NIFTY 100

### Finding Inter-Asset Correlations:
Some of the coefficients we will consider are as follows:
1. Sample correlation
2. Information Coefficient:
    - Pearson's correlation
    - Spearman rank correlation

These statistics will be computed over multiple rolling time windows to capture:

- short-term relationships
- medium-term relationships
- long-term relationships

**Lead–Lag Analysis**

In addition to synchronous correlations, we will explore lead–lag relationships between asset returns to identify delayed dependencies where the movement of one asset systematically precedes another.

### Build pairs:
Once inter-asset similarity measures are obtained, candidate trading pairs will be constructed.

An initial pairing method will be based on a greedy maximum-correlation matching procedure:

1. Rank all asset pairs by correlation strength.

2. Select the pair with the highest correlation.

3. Remove both assets from the candidate pool.

4. Repeat until no assets remain.

This procedure is equivalent to applying Kruskal-style greedy matching on the correlation graph.

Future extensions may include:

- clustering-based pairing
- cointegration testing

### Expected return and volatility modeling:
The following are the methods we will consider to predict returns and risk of each asset for mean-variance optimization:
- **Classical Methods:** Rolling window Moving Average(MA), Exponentially Weighted Moving Average(EWMA), ARIMA
- **Machine Learning Based:** Random Forest, XGBoost
- **Deep Learning Based:** MLP, WaveNet

These methods will further be extended to spread modeling which is used for cointegration based pairs trading strategies.

### Portfolio Creation:
To start off we will start by using the predicted return and variance to produce the portfolio with theoretically highest sharpe ratio by mean-variance optimization and explore other methods.

### Risk Management:
We will explore various heuristic based exit strategies for risk management and evaluate them with suitable metrics mentioned in the next section.

### Evaluation Metrics/Backtesting:
The strategy will be evaluated using walk-forward backtesting on historical data spanning 2015–2026 (tentative).

Performance will be measured using the following metrics:
1. Return
2. Variance
3. Sharpe Ratio
4. Maximum Drawdown
5. VaR
6. CVaR

Walk-forward testing ensures that models are trained only on historically available information, preventing look-ahead bias.

### Baselines:
The following baselines will be used for comparison:
1. **40-60 portfolio**: 40% in risk free assets and 60% in the index fund
2. **Equal Weights**: Equal weights in all assets
3. **Greedy**: Invest 100% in the portfolio with highest expected return
4. **Market Portfolio**: Invest 100% in the index fund.
