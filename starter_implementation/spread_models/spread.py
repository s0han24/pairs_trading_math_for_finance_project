from .hedge_ratio import estimate_hedge_ratio


def compute_spread(y, x):

    beta = estimate_hedge_ratio(y, x)

    spread = y - beta * x

    return spread, beta
