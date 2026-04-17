from cointegration_based.config.universe import NIFTY100
import matplotlib.pyplot as plt
from backtests.walk_forward_pipeline import run_walk_forward_backtest
from cointegration_based.data.downloader import download_price_data
from prettytable import PrettyTable

from cointegration_based.strategy.pipeline import CointegrationPipeline

def run_backtests_coint(prices, total_capital=100.0, method='engle-granger', corr_method='pearson', n=20, k=5, sort_by_corr=False, zscore_method='simple', pvalue_threshold=0.05, corr_threshold=0.8):
    """
    Run a walk-forward cointegration backtest with configurable pair-selection settings.

    Parameters:
        prices (pd.DataFrame): Price history indexed by date with symbols as columns.
        total_capital (float): Initial capital allocated to the strategy.
        method (str): Cointegration test method (for example, 'engle-granger' or 'johansen').
        corr_method (str): Correlation method used for pre-filtering pairs.
        n (int): Number of candidate pairs retained after correlation filtering.
        k (int): Number of pairs selected for trading after cointegration filtering.
        sort_by_corr (bool): If True, rank qualified pairs by correlation instead of p-value.
        zscore_method (str): Z-score model used by the backtest engine.
        pvalue_threshold (float): Maximum p-value allowed for cointegrated pair selection.
        corr_threshold (float): Minimum correlation threshold for pair pre-filtering.
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
        sort_by_corr=sort_by_corr
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
    n = 50  # Number of pairs to keep after correlation filtering
    k = 5   # Number of pairs to trade after cointegration testing
    ''' 
    The following are to be compared(with both sort_by_corr=False and sort_by_corr=True and zscore_method='simple' and zscore_method='ou'):
    sort_by_corr=False means we take top k pairs based on p-value, sort_by_corr=True means we sort the pairs by correlation after filtering by both correlation and the p-value threshold, and then take top k pairs. This allows us to see the impact of prioritizing correlation among the cointegrated pairs.:
    1. Engle-Granger with Pearson filtering 
    3. Johansen with Pearson filtering 
    '''
    prices = download_price_data(tickers)
    
    print("Starting Walk-Forward Backtest...")
    print('Engle-Granger with Pearson filtering')
    equity_eg_pearson, metrics_eg_pearson = run_backtests_coint(prices, total_capital=100.0, method='engle-granger', corr_method='pearson', n=n, k=k, sort_by_corr=False, zscore_method='simple')
    equity_eg_pearson_sorted, metrics_eg_pearson_sorted = run_backtests_coint(prices, total_capital=100.0, method='engle-granger', corr_method='pearson', n=n, k=k, sort_by_corr=True, zscore_method='simple')
    equity_eg_ou, metrics_eg_ou = run_backtests_coint(prices, total_capital=100.0, method='engle-granger', corr_method='pearson', n=n, k=k, sort_by_corr=False, zscore_method='ou')
    equity_eg_ou_sorted, metrics_eg_ou_sorted = run_backtests_coint(prices, total_capital=100.0, method='engle-granger', corr_method='pearson', n=n, k=k, sort_by_corr=True, zscore_method='ou')

    print('Johansen with Pearson filtering')
    equity_johansen_pearson, metrics_johansen_pearson = run_backtests_coint(prices, total_capital=100.0, method='johansen', corr_method='pearson', n=n, k=k, sort_by_corr=False, zscore_method='simple')
    equity_johansen_pearson_sorted, metrics_johansen_pearson_sorted = run_backtests_coint(prices, total_capital=100.0, method='johansen', corr_method='pearson', n=n, k=k, sort_by_corr=True, zscore_method='simple')
    equity_johansen_ou, metrics_johansen_ou = run_backtests_coint(prices, total_capital=100.0, method='johansen', corr_method='pearson', n=n, k=k, sort_by_corr=False, zscore_method='ou')
    equity_johansen_ou_sorted, metrics_johansen_ou_sorted = run_backtests_coint(prices, total_capital=100.0, method='johansen', corr_method='pearson', n=n, k=k, sort_by_corr=True, zscore_method='ou')


    equity_dict = {
        'EG-Pearson': equity_eg_pearson,
        'EG-Pearson-Sorted': equity_eg_pearson_sorted,
        'Johansen-Pearson': equity_johansen_pearson,
        'Johansen-Pearson-Sorted': equity_johansen_pearson_sorted,
        'EG-Pearson-staticOU': equity_eg_ou,
        'EG-Pearson-staticOU-Sorted': equity_eg_ou_sorted,
        'Johansen-Pearson-staticOU': equity_johansen_ou,
        'Johansen-Pearson-staticOU-Sorted': equity_johansen_ou_sorted
    }
    
    # Print metrics in a table
    table = PrettyTable()
    table.field_names = ["Strategy", "Final Capital", "Max Drawdown", "Sharpe Ratio", "Alpha", "Beta", "Annual Volatility"]
    
    metrics_mapping = {
        'EG-Pearson': metrics_eg_pearson,
        'EG-Pearson-Sorted': metrics_eg_pearson_sorted,
        'Johansen-Pearson': metrics_johansen_pearson,
        'Johansen-Pearson-Sorted': metrics_johansen_pearson_sorted,
        'EG-Pearson-staticOU': metrics_eg_ou,
        'EG-Pearson-staticOU-Sorted': metrics_eg_ou_sorted,
        'Johansen-Pearson-staticOU': metrics_johansen_ou,
        'Johansen-Pearson-staticOU-Sorted': metrics_johansen_ou_sorted
    }
    
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
    print(table)
    
    plot_equity_curves(equity_dict)
    
    print('Walk Forward Backtest Completed.')
