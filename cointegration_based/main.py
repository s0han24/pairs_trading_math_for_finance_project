from config.universe import NIFTY100_small, NIFTY50
from data.downloader import download_prices
from pairs.cointegration import find_cointegrated_pairs, find_top_k_cointegrated_pairs_with_filtering, get_top_k_pairs

from spread_models.spread import compute_spread
from spread_models.zscore import zscore

from strategy.signals import generate_positions
from backtest.metrics import sharpe_ratio, max_drawdown, compute_returns
from backtest.backtesting import backtest_pairs

import matplotlib.pyplot as plt
import numpy as np

prices, prices_ood = download_prices(NIFTY50)

# pairs = find_cointegrated_pairs(prices)

# s1, s2, _ = pairs[0]

# print(f"Selected pair: {s1} and {s2}")

# y = prices_ood[s1]
# x = prices_ood[s2]

# spread, beta = compute_spread(y, x)

# z = zscore(spread)

# position = generate_positions(z)

# equity, returns = compute_returns(y, x, beta, position)

# print('Max drawdown:', max_drawdown(equity))
# print('Sharpe ratio:', sharpe_ratio(returns))

# equity.plot()
# plt.show()

pairs_by_pvalue = find_top_k_cointegrated_pairs_with_filtering(prices, pvalue_threshold=0.05, corr_threshold=0.8, n=20, k=5)
print('='*50)

print("Top 5 pairs by p-value(after filtering by correlation):")
print('-'*50)
for s1, s2, corr, pvalue in pairs_by_pvalue:
    print(f"{s1} and {s2} with p-value: {pvalue}")

backtest_pairs(pairs_by_pvalue, prices_ood)

pairs_by_corr = find_top_k_cointegrated_pairs_with_filtering(prices, pvalue_threshold=0.05, corr_threshold=0.8, n=20, k=5, sort_by_corr=True)

print('='*50)
print("Top 5 pairs by correlation:")
print('-'*50)
for s1, s2, corr, pvalue in pairs_by_corr:
    print(f"{s1} and {s2} with correlation: {corr}")
    
backtest_pairs(pairs_by_corr, prices_ood)
