# This module contains functions for filtering pairs before expensive cointegration tests. This is useful when the universe of stocks is large and we want to reduce the number of pairs to test for cointegration.
from itertools import combinations
import numpy as np
from config.settings import CORR_FUNCTION


corr_functions = {
    'pearson': np.corrcoef,
    'spearman': lambda x: np.corrcoef(np.argsort(x, axis=1)),
    # spearman is a placeholder, implement properly in a separate file if needed
}

def filter_pairs_by_correlation(price_df, threshold=0.8, n=10, corr_func=corr_functions.get(CORR_FUNCTION)):
    corr = corr_func(price_df.T)
    pairs = []
    for i, j in combinations(range(len(price_df.columns)), 2):
        if corr[i, j] > threshold:
            pairs.append((price_df.columns[i], price_df.columns[j], corr[i, j]))
    used = set()
    result = []
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    # for s1, s2, c in pairs:
    #     if s1 not in used and s2 not in used:
    #         result.append((s1, s2, c))
    #         used.add(s1)
    #         used.add(s2)
    # return result[:n]
    return pairs[:n]
        