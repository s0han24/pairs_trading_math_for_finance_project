from backtests.metrics import sharpe_ratio, max_drawdown, compute_returns, compute_alpha_beta, compute_annual_volatility
from cointegration_based.spread_models.spread import compute_spread
from cointegration_based.spread_models.zscore import zscore
from cointegration_based.strategy.signals import generate_positions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def backtest_pairs(pairs, prices_ood, zscore_method='simple', total_capital=100.0, plot=False, entry_threshold=2.0, exit_threshold=0.5, stop_loss_threshold=5.0):
    if not exit_threshold < entry_threshold < stop_loss_threshold:
        raise ValueError("Thresholds must satisfy: exit_threshold < entry_threshold < stop_loss_threshold")
    
    k = len(pairs)
    capital_per_pair = total_capital / k
    
    portfolio_equity_series = None
    
    # For benchmark, equal-weight portfolio of the available data
    benchmark_returns = prices_ood.pct_change().mean(axis=1).fillna(0)
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))

    for i, pair in enumerate(pairs):
        s1 = pair[0]
        s2 = pair[1]
        
        y = prices_ood[s1]
        x = prices_ood[s2]

        spread, beta_hr = compute_spread(y, x)

        z = zscore(spread, method=zscore_method)

        position = generate_positions(z, entry=entry_threshold, exit=exit_threshold, stop_loss=stop_loss_threshold)

        # compute_returns provides base 1-unit equity and returns
        _, strategy_returns = compute_returns(y, x, beta_hr, position)
        
        # Calculate strategy equity from allocated capital
        pair_equity = capital_per_pair * (1 + strategy_returns).cumprod()
        
        if plot:
            pair_equity.plot(ax=ax, label=f'Pair {i+1}: {s1}-{s2}', alpha=0.5)
        
        if portfolio_equity_series is None:
            portfolio_equity_series = pair_equity.copy()
        else:
            portfolio_equity_series += pair_equity

    if plot:
        portfolio_equity_series.plot(ax=ax, label='Total Portfolio', color='black', linewidth=2)
        ax.legend()
        plt.title('Out-of-Sample Equity')
        plt.show()
    
    # Portfolio Metrics
    portfolio_returns = portfolio_equity_series.pct_change().dropna()
    benchmark_returns = benchmark_returns.reindex(portfolio_returns.index)
    
    mdd = max_drawdown(portfolio_equity_series)
    sr = sharpe_ratio(portfolio_returns)
    alpha, beta = compute_alpha_beta(portfolio_returns, benchmark_returns)
    vol = compute_annual_volatility(portfolio_returns)
    
    # Print metrics in a table
    print('='*50)
    print("Portfolio Performance Metrics")
    print('-'*50)
    print(f"Total Initial Capital:  ${total_capital:.2f}")
    print(f"Number of Pairs:        {k}")
    print(f"Max Drawdown:           {mdd:.2%}")
    print(f"Sharpe Ratio:           {sr:.2f}")
    print(f"Alpha (Annual):         {alpha:.2%}")
    print(f"Beta:                   {beta:.2f}")
    print(f"Annual Volatility:      {vol:.2%}")
    print('='*50)
    
    return portfolio_equity_series, portfolio_returns

