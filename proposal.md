# Pairs Trading Framework

### Problem Setup:
A pairs trading strategy identifies two co-moving assets that have temporarily diverged in price. The strategy is to go long on the underperformer and short on the overperformer, betting on the mean reverting property of the assets. Another component of the strategy is risk management to cap losses.

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

- The correlations will be taken over different time window sizes to consider short term, long term and anything in between.
- Further, we will use lead-lag based methods to capture delayed correlation

### Build pairs:
A method to start off with is a Kruskal's algorithm based correlation which is as follows. Start pairing the assets by highest correlation, remove the assets which have been paired and continue. 

### Expected return and volatility prediction:
The following are the methods we will consider:
- **Classical Methods:** Rolling window Moving Average(MA), Exponentially Weighted Moving Average(EWMA), ARIMA
- **Machine Learning Based:** Random Forest, XGBoost
- **Deep Learning Based:** MLP, WaveNet, WaveRNN

### Portfolio Creation:
To start off we will start by using the predicted return and variance to produce the portfolio with highest sharpe ratio and explore other methods.

### Evaluation Metrics/Backtesting
The following evaluation metrics will be used:
1. Return
2. Variance
3. Sharpe Ratio
4. Maximum Drawdown
5. VaR
6. CVaR

The evaluation will be performed via walk-forward testing on data from 2015-2026(tentatively).

### Baselines:
The following baselines will be used for comparison:
1. **40-60 portfolio**: 40% in risk free assets and 60% in the index fund
2. **Equal Weights**: Equal weights in all assets
3. **Greedy**: Invest 100% in the portfolio with highest expected return
4. **Market Portfolio**: Invest 100% in the index fund.
