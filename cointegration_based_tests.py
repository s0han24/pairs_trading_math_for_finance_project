from cointegration_based.config.universe import NIFTY100
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cointegration_based.strategy.pipeline import PairsTradingPipeline
from cointegration_based.backtest.metrics import sharpe_ratio, max_drawdown
from cointegration_based.backtest.backtesting import compute_alpha_beta, compute_annual_volatility
from prettytable import PrettyTable

def run_walk_forward_backtest(prices, total_capital=100.0, method='engle-granger', corr_method='pearson', n=20, k=5, sort_by_corr=False):
    
    # 2 years (approx 504 trading days) in sample, 2 years out sample
    # Walk forward windows:
    # 1: 2014-2016 in, 2016-2018 out
    # 2: 2016-2018 in, 2018-2020 out
    # 3: 2018-2020 in, 2020-2022 out
    # 4: 2020-2022 in, 2022-2024 out
    # 5: 2022-2024 in, 2024-2026 out
    
    windows = [
        ("2014-01-01", "2016-01-01", "2018-01-01"),
        ("2016-01-01", "2018-01-01", "2020-01-01"),
        ("2018-01-01", "2020-01-01", "2022-01-01"),
        ("2020-01-01", "2022-01-01", "2024-01-01"),
        ("2022-01-01", "2024-01-01", "2026-01-01")
    ]
    
    all_equity = []
    
    current_capital = total_capital
    
    benchmark_equity = [total_capital]
    benchmark_returns_list = []
    equity_timestamps = []
    
    for start_in, end_in_start_ood, end_ood in windows:
        print(f"\n--- Walk Forward Step: Train [{start_in} to {end_in_start_ood}], Test [{end_in_start_ood} to {end_ood}] ---")
        
        train_prices = prices.loc[start_in:end_in_start_ood]
        test_prices = prices.loc[end_in_start_ood:end_ood]
        
        pipeline = PairsTradingPipeline(
            coint_test_method=method, 
            corr_method=corr_method,
            total_capital=current_capital,
            pvalue_threshold=0.05,
            corr_threshold=0.8,
            n=n,
            k=k
        )
        
        # Fit on training data
        selected_pairs = pipeline.fit(train_prices, sort_by_corr=sort_by_corr)
        print(f"Selected {len(selected_pairs)} pairs.")
        for s1, s2, corr, pvalue in selected_pairs:
            print(f"  {s1}-{s2} (P-value: {pvalue:.4f}, Corr: {corr:.2f})")
            
        # Benchmark
        bench_ret = test_prices.pct_change().mean(axis=1).fillna(0)
        benchmark_returns_list.append(bench_ret)
        
        # Backtest on test data
        if not selected_pairs:
            print("No pairs found. Holding cash.")
            # Equity remains flat
            cash_eq = pd.Series(current_capital, index=test_prices.index)
            all_equity.append(cash_eq)
        else:
            equity_series, _ = pipeline.backtest(test_prices, plot=False)
            all_equity.append(equity_series)
            current_capital = equity_series.iloc[-1]
            
    # Combine walk-forward pieces 
    # Drop duplicates at boundaries
    total_portfolio_equity = pd.concat(all_equity)
    total_portfolio_equity = total_portfolio_equity[~total_portfolio_equity.index.duplicated(keep='last')]
    
    total_benchmark_ret = pd.concat(benchmark_returns_list)
    total_benchmark_ret = total_benchmark_ret[~total_benchmark_ret.index.duplicated(keep='last')]
    
    # Compute overall metrics
    returns = total_portfolio_equity.pct_change().dropna()
    bench_returns = total_benchmark_ret.reindex(returns.index).fillna(0)
    
    mdd = max_drawdown(total_portfolio_equity)
    sr = sharpe_ratio(returns)
    alpha, beta = compute_alpha_beta(returns, bench_returns)
    vol = compute_annual_volatility(returns)
    
    metrics_dict = {
        'final_capital': total_portfolio_equity.iloc[-1],
        'max_drawdown': mdd,
        'sharpe_ratio': sr,
        'alpha': alpha,
        'beta': beta,
        'annual_volatility': vol
    }
    
    return total_portfolio_equity, metrics_dict

def plot_equity_curves(equity_dict):
    plt.figure(figsize=(12, 8))
    for label, equity in equity_dict.items():
        plt.plot(equity.index, equity.values, label=label)
    plt.title("Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()
    
def download_price_data(tickers, start="2014-01-01", end="2026-01-01"):
    print("Downloading price data...")
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    prices = prices.dropna(axis=1)
    return prices

if __name__ == "__main__":
    tickers = NIFTY100
    n = 50  # Number of pairs to keep after correlation filtering
    k = 5   # Number of pairs to trade after cointegration testing
    ''' 
    The following are to be compared(with both sort_by_corr=False and sort_by_corr=True):
    sort_by_corr=False means we take top k pairs based on p-value, sort_by_corr=True means we sort the pairs by correlation after filtering by both correlation and the p-value threshold, and then take top k pairs. This allows us to see the impact of prioritizing correlation among the cointegrated pairs.:
    1. Engle-Granger with Pearson filtering 
    3. Johansen with Pearson filtering 
    '''
    prices = download_price_data(tickers)
    
    print("Starting Walk-Forward Backtest...")
    print('Engle-Granger with Pearson filtering')
    equity_eg_pearson, metrics_eg_pearson = run_walk_forward_backtest(prices, method='engle-granger', corr_method='pearson', n=n, k=k, sort_by_corr=False)
    equity_eg_pearson_sorted, metrics_eg_pearson_sorted = run_walk_forward_backtest(prices, method='engle-granger', corr_method='pearson', n=n, k=k, sort_by_corr=True)
    
    print('Johansen with Pearson filtering')

    equity_johansen_pearson, metrics_johansen_pearson = run_walk_forward_backtest(prices, method='johansen', corr_method='pearson', n=n, k=k, sort_by_corr=False)
    equity_johansen_pearson_sorted, metrics_johansen_pearson_sorted = run_walk_forward_backtest(prices, method='johansen', corr_method='pearson', n=n, k=k, sort_by_corr=True)
    
    
    equity_dict = {
        'EG-Pearson': equity_eg_pearson,
        'EG-Pearson-Sorted': equity_eg_pearson_sorted,
        'Johansen-Pearson': equity_johansen_pearson,
        'Johansen-Pearson-Sorted': equity_johansen_pearson_sorted,
    }
    
    # Print metrics in a table
    table = PrettyTable()
    table.field_names = ["Strategy", "Final Capital", "Max Drawdown", "Sharpe Ratio", "Alpha", "Beta", "Annual Volatility"]
    for label, equity in equity_dict.items():
        metrics = None
        if label == 'EG-Pearson':
            metrics = metrics_eg_pearson
        elif label == 'EG-Pearson-Sorted':
            metrics = metrics_eg_pearson_sorted
        elif label == 'Johansen-Pearson':
            metrics = metrics_johansen_pearson
        elif label == 'Johansen-Pearson-Sorted':
            metrics = metrics_johansen_pearson_sorted
        
        table.add_row([
            label, 
            f"{metrics['final_capital']:.2f}", 
            f"{metrics['max_drawdown']:.2%}", 
            f"{metrics['sharpe_ratio']:.2f}", 
            f"{metrics['alpha']:.4f}", 
            f"{metrics['beta']:.4f}", 
            f"{metrics['annual_volatility']:.2%}"
        ])
    print(table)
    
    plot_equity_curves(equity_dict)
    
    print('Walk Forward Backtest Completed.')
