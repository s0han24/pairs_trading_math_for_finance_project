import numpy as np

def pearson_correlation(prices):
    return np.corrcoef(prices.T)
