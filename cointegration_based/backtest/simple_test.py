# Not a real backtest, just a simple test to see if the returns are being computed correctly

def compute_returns(y, x, beta, position):

    returns = y.pct_change() - beta * x.pct_change()

    strategy_returns = position[:-1] * returns[1:]

    equity = (1 + strategy_returns).cumprod()

    return equity, strategy_returns
