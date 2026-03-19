def sharpe_ratio(returns, risk_free_rate=0.0, frequency=252):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() * (frequency ** 0.5)

def max_drawdown(equity):

    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    return drawdown.min()

def compute_returns(y, x, beta, position):

    returns = y.pct_change() - beta * x.pct_change()

    strategy_returns = position[:-1] * returns[1:]

    equity = (1 + strategy_returns).cumprod()

    return equity, strategy_returns
