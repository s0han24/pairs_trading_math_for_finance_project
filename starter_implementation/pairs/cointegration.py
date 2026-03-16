from itertools import combinations
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(price_df, pvalue_threshold=0.05):

    pairs = []

    for s1, s2 in combinations(price_df.columns, 2):

        score, pvalue, _ = coint(price_df[s1], price_df[s2])

        if pvalue < pvalue_threshold:
            pairs.append((s1, s2, pvalue))

    pairs = sorted(pairs, key=lambda x: x[2])

    return pairs
