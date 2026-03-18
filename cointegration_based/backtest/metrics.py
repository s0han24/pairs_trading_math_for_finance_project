def sharpe_ratio(returns, risk_free_rate=0.0, frequency=252):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() * (frequency ** 0.5)

def max_drawdown(equity):

    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    return drawdown.min()
