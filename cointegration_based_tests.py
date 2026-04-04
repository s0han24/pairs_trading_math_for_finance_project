from cointegration_based.config.universe import NIFTY100_small, NIFTY50
from cointegration_based.data.downloader import download_prices
from cointegration_based.pairs.cointegration import find_cointegrated_pairs, find_top_k_cointegrated_pairs_with_filtering, get_top_k_pairs

from cointegration_based.spread_models.spread import compute_spread
from cointegration_based.spread_models.zscore import zscore

from cointegration_based.strategy.signals import generate_positions
from cointegration_based.backtest.metrics import sharpe_ratio, max_drawdown, compute_returns
from cointegration_based.backtest.backtesting import backtest_pairs

import matplotlib.pyplot as plt
import numpy as np



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

def test_coint_methods(prices, prices_ood):
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

# During COVID crash, many stocks had very high correlation but were not truly cointegrated. This test checks if our filtering can identify good pairs in such a scenario.

prices_covid, prices_ood_covid = download_prices(NIFTY50, start_in="2018-01-01", end_in="2020-01-01", start_ood="2020-01-01", end_ood="2022-01-01")
print('Testing on COVID data...')
test_coint_methods(prices_covid, prices_ood_covid)

# Pre-COVID period (2014-2016) had more stable relationships. This test checks if we can find good pairs and achieve better performance compared to the COVID period.
# However, many stocks did not exist in 2014, so we may have fewer pairs to work with. This tests the robustness of our method in a different market regime.

prices_pre_covid, prices_ood_pre_covid = download_prices(NIFTY50, start_in="2014-01-01", end_in="2016-01-01", start_ood="2016-01-01", end_ood="2018-01-01")
print('Testing on pre-COVID data...')
test_coint_methods(prices_pre_covid, prices_ood_pre_covid)

# Post-COVID period (2022-2024) had a mix of recovery and volatility. This test checks if our method can adapt to changing market conditions and still find good pairs.
prices_post_covid, prices_ood_post_covid = download_prices(NIFTY50, start_in="2022-01-01", end_in="2024-01-01", start_ood="2024-01-01", end_ood="2026-01-01")
print('Testing on post-COVID data...')
test_coint_methods(prices_post_covid, prices_ood_post_covid)
