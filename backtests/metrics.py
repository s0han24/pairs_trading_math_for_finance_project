import numpy as np

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

def compute_alpha_beta(strategy_returns, benchmark_returns, risk_free_rate=0.0, frequency=252):
    excess_strategy = strategy_returns - risk_free_rate / frequency
    excess_benchmark = benchmark_returns - risk_free_rate / frequency
    
    # Calculate beta
    cov = np.cov(excess_strategy, excess_benchmark)[0, 1]
    var_bench = np.var(excess_benchmark)
    beta = cov / var_bench if var_bench != 0 else 0.0
    
    # Calculate annualized alpha
    alpha_daily = np.mean(excess_strategy) - beta * np.mean(excess_benchmark)
    alpha = alpha_daily * frequency
    
    return alpha, beta

def compute_annual_volatility(returns, frequency=252):
    return returns.std() * np.sqrt(frequency)
