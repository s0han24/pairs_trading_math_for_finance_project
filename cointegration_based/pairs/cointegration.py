from itertools import combinations
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np

from .filtering import filter_pairs_by_correlation
from cointegration_based.spread_models.spread import compute_spread
from cointegration_based.spread_models.zscore import zscore
from cointegration_based.strategy.signals import generate_positions


def _get_johansen_pvalue(johansen_result):
    """
    Extracts the p-value from the Johansen test result.
    It returns the p-value for the trace statistic with 0 cointegrating equations.
    """
    # The trace statistic and critical values are in the lr1 and cvt attributes
    trace_stat = johansen_result.lr1[0]
    crit_values = johansen_result.cvt[0]  # 90%, 95%, 99%

    if trace_stat > crit_values[2]:  # more than 99% confidence
        return 0.00
    elif trace_stat > crit_values[1]:  # more than 95% confidence
        return 0.05
    elif trace_stat > crit_values[0]:  # more than 90% confidence
        return 0.10
    else:
        return 1.0


def find_cointegrated_pairs(price_df, pvalue_threshold=0.05, coint_test_method='engle-granger'):
    pairs = []
    for s1, s2 in combinations(price_df.columns, 2):
        if coint_test_method == 'engle-granger':
            _, pvalue, _ = coint(price_df[s1], price_df[s2])
        elif coint_test_method == 'johansen':
            df = price_df[[s1, s2]]
            johansen_result = coint_johansen(df, det_order=0, k_ar_diff=1)
            pvalue = _get_johansen_pvalue(johansen_result)
        else:
            raise ValueError("Invalid cointegration test method in settings.")

        if pvalue < pvalue_threshold:
            pairs.append((s1, s2, pvalue))

    pairs = sorted(pairs, key=lambda x: x[2])
    return pairs


def find_top_k_cointegrated_pairs_with_filtering(price_df, pvalue_threshold=0.05, corr_threshold=0.8, n=10, k=5, sort_by='corr', coint_test_method='engle-granger', corr_method='pearson'):
    filtered_pairs = filter_pairs_by_correlation(
        price_df, threshold=corr_threshold, n=n, corr_method=corr_method)
    pairs = []
    for s1, s2, corr in filtered_pairs:
        if coint_test_method == 'engle-granger':
            _, pvalue, _ = coint(price_df[s1], price_df[s2])
        elif coint_test_method == 'johansen':
            df = price_df[[s1, s2]]
            johansen_result = coint_johansen(df, det_order=0, k_ar_diff=1)
            pvalue = _get_johansen_pvalue(johansen_result)
        else:
            raise ValueError("Invalid cointegration test method in settings.")

        if pvalue < pvalue_threshold:
            pairs.append((s1, s2, corr, pvalue))

    if sort_by == 'corr':
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    elif sort_by == 'sr':
        return get_top_k_pairs_by_sharpe(pairs, price_df, k=k)
    else:
        pairs = sorted(pairs, key=lambda x: x[3], reverse=False)

    return pairs[:k]

def get_top_k_pairs_by_sharpe(pairs, price_df, k=5, window=60, entry=2.0, exit=0.5, stop_loss=5.0):
    """Use in-sample Sharpe ratio to select top k pairs from the list of cointegrated pairs."""
    pair_sharpes = []
    for s1, s2, corr, pvalue in pairs:
        # Simulate pairs trading on the in-sample data
        spread, _ = compute_spread(price_df[s1], price_df[s2])
        z = zscore(spread, window=window, method='simple')
        positions = generate_positions(z, entry=entry, exit=exit, stop_loss=stop_loss)
        
        # Calculate daily returns of the strategy on the spread
        positions_series = pd.Series(positions, index=spread.index)
        spread_diff = spread.diff()
        
        # Shift position by 1 because the return at t is based on the position established at the end of t-1
        strategy_returns = positions_series.shift(1) * spread_diff
        strategy_returns = strategy_returns.dropna()
        
        std = strategy_returns.std()
        if std < 1e-10 or np.isnan(std):
            sharpe = 0.0
        else:
            sharpe = strategy_returns.mean() / std * np.sqrt(252)
        pair_sharpes.append((s1, s2, corr, pvalue, sharpe))
    pair_sharpes = sorted(pair_sharpes, key=lambda x: x[4], reverse=True)
    selected_pairs = [(s1, s2, corr, pvalue) for s1, s2, corr, pvalue, sharpe in pair_sharpes[:k]]
    
    return selected_pairs
