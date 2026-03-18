from config.universe import NIFTY100
from data.downloader import download_prices
from pairs.cointegration import find_cointegrated_pairs

from spread_models.spread import compute_spread
from spread_models.zscore import zscore

from strategy.signals import generate_positions
from backtest.backtester import compute_returns

import matplotlib.pyplot as plt

from backtest.metrics import sharpe_ratio, max_drawdown


prices = download_prices(NIFTY100)

pairs = find_cointegrated_pairs(prices)

s1, s2, _ = pairs[0]

print(f"Selected pair: {s1} and {s2}")

y = prices[s1]
x = prices[s2]

spread, beta = compute_spread(y, x)

z = zscore(spread)

position = generate_positions(z)

equity, returns = compute_returns(y, x, beta, position)

print('Max drawdown:', max_drawdown(equity))
print('Sharpe ratio:', sharpe_ratio(returns))

equity.plot()
plt.show()
