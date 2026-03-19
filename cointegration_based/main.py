from config.universe import NIFTY100_small, NIFTY50
from data.downloader import download_prices
from pairs.cointegration import find_cointegrated_pairs, find_top_k_cointegrated_pairs_with_filtering, get_top_k_pairs

from spread_models.spread import compute_spread
from spread_models.zscore import zscore

from strategy.signals import generate_positions
from backtest.simple_test import compute_returns

import matplotlib.pyplot as plt
import numpy as np

from backtest.metrics import sharpe_ratio, max_drawdown


prices = download_prices(NIFTY50)

# pairs = find_cointegrated_pairs(prices)

# s1, s2, _ = pairs[0]

# print(f"Selected pair: {s1} and {s2}")

# y = prices[s1]
# x = prices[s2]

# spread, beta = compute_spread(y, x)

# z = zscore(spread)

# position = generate_positions(z)

# equity, returns = compute_returns(y, x, beta, position)

# print('Max drawdown:', max_drawdown(equity))
# print('Sharpe ratio:', sharpe_ratio(returns))

# equity.plot()
# plt.show()

pairs = find_top_k_cointegrated_pairs_with_filtering(prices, pvalue_threshold=0.05, corr_threshold=0.8, n=20, k=5)

print("Top 5 pairs:")
for s1, s2, pvalue in pairs:
    print(f"{s1} and {s2} with p-value: {pvalue}")
    
equity_curves = []
returns_list = []
# make portfolio of top 5 pairs and backtest it
for i, (s1, s2, corr, pvalue) in enumerate(pairs):

    print(f"Backtesting pair: {s1} and {s2}")

    y = prices[s1]
    x = prices[s2]

    spread, beta = compute_spread(y, x)

    z = zscore(spread)

    position = generate_positions(z)

    equity, returns = compute_returns(y, x, beta, position)
    
    print('Max drawdown:', max_drawdown(equity))
    print('Sharpe ratio:', sharpe_ratio(returns))
    equity.plot(label=f'Pair {i+1}')
    equity_curves.append(equity)
    returns_list.append(returns)
plt.legend()
plt.show()
