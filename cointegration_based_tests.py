import itertools
from config.universe import NIFTY100
import matplotlib.pyplot as plt
from backtests.walk_forward_pipeline import run_walk_forward_backtest
from data.downloader import download_price_data
from prettytable import PrettyTable

from cointegration_based.strategy.pipeline import CointegrationPipeline

def run_backtests_coint(prices, total_capital=100.0, method='engle-granger', corr_method='pearson', n=20, k=5, sort_by='pval', zscore_method='simple', pvalue_threshold=0.05, corr_threshold=0.8, entry_threshold=2.0, exit_threshold=0.5, stop_loss_threshold=5.0):
    """
    Run a walk-forward cointegration backtest with configurable pair-selection settings.

    Parameters:
        prices (pd.DataFrame): Price history indexed by date with symbols as columns.
        total_capital (float): Initial capital allocated to the strategy.
        method (str): Cointegration test method (for example, 'engle-granger' or 'johansen').
        corr_method (str): Correlation method used for pre-filtering pairs.
        n (int): Number of candidate pairs retained after correlation filtering.
        k (int): Number of pairs selected for trading after cointegration filtering.
        sort_by (str): Criteria to sort qualified pairs ('pval', 'corr', or 'sr').
        zscore_method (str): Z-score model used by the backtest engine.
        pvalue_threshold (float): Maximum p-value allowed for cointegrated pair selection.
        corr_threshold (float): Minimum correlation threshold for pair pre-filtering.
        entry_threshold (float): Z-score threshold for entering a position.
        exit_threshold (float): Z-score threshold for exiting a position.
        stop_loss_threshold (float): Z-score threshold for triggering a stop loss.
    """
    
    pipeline = CointegrationPipeline(
        coint_test_method=method,
        corr_method=corr_method,
        pvalue_threshold=pvalue_threshold,
        corr_threshold=corr_threshold,
        n=n,
        k=k,
        total_capital=total_capital,
        zscore_method=zscore_method,
        sort_by=sort_by,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss_threshold=stop_loss_threshold
    )
    return run_walk_forward_backtest(prices, pipeline=pipeline)

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

if __name__ == "__main__":
    tickers = NIFTY100
    n = 100  # Number of pairs to keep after correlation filtering
    k = 5   # Number of pairs to trade after cointegration testing
    ''' 
    The following are to be compared(with both sort_by='pval' and sort_by='corr' and zscore_method='simple' and zscore_method='ou'):
    sort_by='pval' means we take top k pairs based on p-value, sort_by='corr' means we sort the pairs by correlation after filtering by both correlation and the p-value threshold, and then take top k pairs. This allows us to see the impact of prioritizing correlation among the cointegrated pairs.:
    1. Engle-Granger with Pearson filtering 
    3. Johansen with Pearson filtering 
    '''
    prices = download_price_data(tickers)
    
    print("Starting Walk-Forward Backtest with Grid Search...")
    
    methods = ['engle-granger', 'johansen']
    # sort_bys = ['pval', 'corr', 'sr']
    # sort_bys = ['sr', 'pval'] # pval and corr are very similar in performance, so skipping corr
    sort_bys = ['sr'] # SR performs best among the 3, so only showing that in the final results table for better clarity.
    zscore_methods = ['simple', 'ou']
    stop_losses = [float('inf'), 5.0]
    
    equity_dict = {}
    metrics_mapping = {}
    
    grid = list(itertools.product(methods, sort_bys, zscore_methods, stop_losses))
    
    for method, sort_by, zscore_method, stop_loss in grid:
        label_parts = []
        label_parts.append('EG' if method == 'engle-granger' else 'Johansen')
        label_parts.append('Pearson')
        
        if zscore_method == 'ou':
            label_parts.append('staticOU')
            
        if sort_by == 'corr':
            label_parts.append('Sorted')
        elif sort_by == 'sr':
            label_parts.append('SR-Sorted')
            
        if stop_loss == 5.0:
            label_parts.append('StopLoss')

        label = "-".join(label_parts)
        print(f"Running backtest for: {label}")
        
        equity, metrics = run_backtests_coint(
            prices, 
            total_capital=100.0, 
            method=method, 
            corr_method='pearson', 
            n=n, 
            k=k, 
            sort_by=sort_by, 
            zscore_method=zscore_method, 
            entry_threshold=2.0, 
            exit_threshold=0.5, 
            stop_loss_threshold=stop_loss
        )
        
        equity_dict[label] = equity
        metrics_mapping[label] = metrics
    
    table = PrettyTable()
    table.field_names = ["Strategy", "Final Capital", "Max Drawdown", "Sharpe Ratio", "Alpha", "Beta", "Annual Volatility"]
    
    for label, equity in equity_dict.items():
        metrics = metrics_mapping.get(label)
        
        table.add_row([
            label, 
            f"{metrics['final_capital']:.2f}", 
            f"{metrics['max_drawdown']:.2%}", 
            f"{metrics['sharpe_ratio']:.2f}", 
            f"{metrics['alpha']:.4f}", 
            f"{metrics['beta']:.4f}", 
            f"{metrics['annual_volatility']:.2%}"
        ])
    # Sort by final capital descending before building table
    sorted_labels = sorted(
        equity_dict.keys(),
        key=lambda lbl: metrics_mapping[lbl]['final_capital'],
        reverse=True
    )
    
    for label in sorted_labels:
        metrics = metrics_mapping.get(label)
        
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
