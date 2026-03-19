from .metrics import sharpe_ratio, max_drawdown, compute_returns
from spread_models.spread import compute_spread
from spread_models.zscore import zscore
from strategy.signals import generate_positions
import matplotlib.pyplot as plt
import numpy as np

def backtest_pairs(pairs, prices_ood):
    for i, (s1, s2, corr, pvalue) in enumerate(pairs):

        print(f"Backtesting pair: {s1} and {s2}")

        # Backtest on OOD data
        y = prices_ood[s1]
        x = prices_ood[s2]

        spread, beta = compute_spread(y, x)

        z = zscore(spread)

        position = generate_positions(z)

        equity, returns = compute_returns(y, x, beta, position)
        
        print('Max drawdown:', max_drawdown(equity))
        print('Sharpe ratio:', sharpe_ratio(returns))
        equity.plot(label=f'Pair {i+1}')
    plt.legend()
    plt.show()

