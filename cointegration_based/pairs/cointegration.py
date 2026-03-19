from itertools import combinations
from statsmodels.tsa.stattools import coint
from .filtering import filter_pairs_by_correlation


def find_cointegrated_pairs(price_df, pvalue_threshold=0.05):

    pairs = []

    for s1, s2 in combinations(price_df.columns, 2):

        score, pvalue, _ = coint(price_df[s1], price_df[s2])

        if pvalue > pvalue_threshold:
            pairs.append((s1, s2, pvalue))

    pairs = sorted(pairs, key=lambda x: x[2])

    return pairs

def find_top_k_cointegrated_pairs_with_filtering(price_df, pvalue_threshold=0.05, corr_threshold=0.8, n=10, k=5, sort_by_corr=False):
    filtered_pairs = filter_pairs_by_correlation(price_df, threshold=corr_threshold, n=n)

    pairs = []

    for s1, s2, corr in filtered_pairs:

        score, pvalue, _ = coint(price_df[s1], price_df[s2])

        if pvalue > pvalue_threshold:
            pairs.append((s1, s2, corr, pvalue))
    if sort_by_corr:
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    else:
        pairs = sorted(pairs, key=lambda x: x[3], reverse=True)

    return pairs[:k]

def get_top_k_pairs(pairs, k=5):
    result = []
    included = set()
    i = 0
    while(len(result) < k and len(included) < 2 * k and i < len(pairs)):
        s1, s2, pvalue = pairs[i]
        if s1 not in included and s2 not in included:
            result.append((s1, s2, pvalue))
            included.add(s1)
            included.add(s2)
        i += 1
    return result
