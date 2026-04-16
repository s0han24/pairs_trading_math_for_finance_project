import pandas as pd
from cointegration_based.pairs.cointegration import find_top_k_cointegrated_pairs_with_filtering
from cointegration_based.backtest.backtesting import backtest_pairs

class PairsTradingPipeline:
    def __init__(self, 
                 coint_test_method='engle-granger', 
                 corr_method='pearson',
                 zscore_method='simple',
                 pvalue_threshold=0.05, 
                 corr_threshold=0.8, 
                 n=20, 
                 k=5, 
                 total_capital=100.0):
        self.coint_test_method = coint_test_method
        self.corr_method = corr_method
        self.pvalue_threshold = pvalue_threshold
        self.corr_threshold = corr_threshold
        self.n = n
        self.k = k
        self.total_capital = total_capital
        self.zscore_method = zscore_method

        self.selected_pairs = []

    def fit(self, train_prices, sort_by_corr=False):
        self.selected_pairs = find_top_k_cointegrated_pairs_with_filtering(
            train_prices, 
            pvalue_threshold=self.pvalue_threshold, 
            corr_threshold=self.corr_threshold, 
            n=self.n, 
            k=self.k, 
            sort_by_corr=sort_by_corr,
            coint_test_method=self.coint_test_method,
            corr_method=self.corr_method
        )
        return self.selected_pairs

    def backtest(self, test_prices, plot=False):
        if not self.selected_pairs:
            print("No pairs selected. Please run fit() first, or try adjusting thresholds.")
            return None, None
            
        return backtest_pairs(self.selected_pairs, test_prices, total_capital=self.total_capital, plot=plot, zscore_method=self.zscore_method)
