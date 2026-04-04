import numpy as np
from correlation_statistics.hsic import hsic_gam

def pearson_correlation(prices):
    return np.corrcoef(prices.T)

# Assumes i.i.d. samples which is not true for financial data, but we can still use it as a heuristic for filtering pairs before cointegration testing.
def hsic(prices):
    return hsic_gam(prices, prices)[0] # test statistic, we can ignore the threshold for now since we will be comparing relative values for filtering pairs

# We can add more correlation functions here in the future if needed, and make it configurable in settings.py
