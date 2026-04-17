import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtests.metrics import (
    sharpe_ratio,
    max_drawdown,
    compute_alpha_beta,
    compute_annual_volatility,
)
from data.downloader import download_price_data
from config.universe import NIFTY100

def _compute_metrics(equity: pd.Series,
                     benchmark_returns: pd.Series,
                     total_capital: float,
                     frequency: int = 252) -> dict:
    """
    Compute performance metrics for an equity curve.
    """

    returns = equity.pct_change().dropna()
    bench = benchmark_returns.reindex(returns.index).fillna(0)
    alpha, beta = compute_alpha_beta(returns, bench, frequency=frequency)

    return {
        "final_capital":     equity.iloc[-1],
        "total_return":      (equity.iloc[-1] / total_capital) - 1,
        "max_drawdown":      max_drawdown(equity),
        "sharpe_ratio":      sharpe_ratio(returns),
        "alpha":             alpha,
        "beta":              beta,
        "annual_volatility": compute_annual_volatility(returns),
    }


def _print_metrics(name: str, metrics: dict) -> None:
    print("=" * 50)
    print(f"  {name}")
    print("-" * 50)
    print(f"  Final Capital:      ${metrics['final_capital']:>10.2f}")
    print(f"  Total Return:       {metrics['total_return']:>10.2%}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:>10.2%}")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Alpha (Annual):     {metrics['alpha']:>10.4f}")
    print(f"  Beta:               {metrics['beta']:>10.4f}")
    print(f"  Annual Volatility:  {metrics['annual_volatility']:>10.2%}")
    print("=" * 50)

# Baseline 1 : 40-60  (40% risk-free + 60% MARKET PORTFOLIO, daily rebalanced)
class FortysixtBaseline:
    """
    40% cash at the risk-free rate + 60% Market Portfolio (^NSEI), daily rebalanced.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        self.daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    def fit(self, train_prices: pd.DataFrame) -> None:
        pass

    def backtest(self,
                 test_prices: pd.DataFrame,
                 benchmark_returns: pd.Series,          
                 total_capital: float = 100.0):
        
        # 60% allocated to the Market Portfolio (which is our benchmark_returns)
        market_returns = benchmark_returns

        cash       = 0.40 * total_capital
        equity_cap = 0.60 * total_capital

        n = len(market_returns)
        cash_equity = cash * (1 + self.daily_rf) ** np.arange(n)
        cash_equity = pd.Series(cash_equity, index=market_returns.index)

        equity_part = equity_cap * (1 + market_returns).cumprod()
        equity      = cash_equity + equity_part

        returns = equity.pct_change().fillna(0)
        metrics = _compute_metrics(equity, benchmark_returns, total_capital)
        return equity, returns, metrics

# Baseline 2 : Equal Weight (daily rebalanced)
class EqualWeightBaseline:
    """
    Daily rebalanced equal-weight portfolio.
    """

    def fit(self, train_prices: pd.DataFrame) -> None:
        pass

    def backtest(self,
                 test_prices: pd.DataFrame,
                 benchmark_returns: pd.Series,          
                 total_capital: float = 100.0):

        returns = test_prices.pct_change().fillna(0).mean(axis=1)
        equity  = total_capital * (1 + returns).cumprod()

        final_returns = equity.pct_change().fillna(0)
        metrics = _compute_metrics(equity, benchmark_returns, total_capital)
        return equity, final_returns, metrics

# Baseline 3 : Equal Weight Buy & Hold
class EqualWeightBuyHoldBaseline:
    """
    Equal-weight BUY-AND-HOLD. Weights are set once at t=0 and allowed to drift.
    """

    def fit(self, train_prices: pd.DataFrame) -> None:
        pass

    def backtest(self,
                 test_prices: pd.DataFrame,
                 benchmark_returns: pd.Series,          
                 total_capital: float = 100.0):

        n                 = test_prices.shape[1]
        capital_per_stock = total_capital / n

        stock_equity = capital_per_stock * (test_prices / test_prices.iloc[0])
        equity       = stock_equity.sum(axis=1)

        # FIX: fillna(0) instead of dropna()
        returns  = equity.pct_change().fillna(0)
        metrics  = _compute_metrics(equity, benchmark_returns, total_capital)
        return equity, returns, metrics

# Baseline 4 : Greedy
class GreedyBaseline:
    """
    Invest 100% in the stock with the highest total cumulative return in-sample.
    """

    def __init__(self):
        self.best_stock = None

    def fit(self, train_prices: pd.DataFrame) -> None:
        cum_gain = (train_prices.iloc[-1] / train_prices.iloc[0]) - 1
        self.best_stock = cum_gain.idxmax()
        print(f"[Greedy] Selected: {self.best_stock}  "
              f"(in-sample cumulative gain: {cum_gain[self.best_stock]:+.2%})")

    def backtest(self,
                 test_prices: pd.DataFrame,
                 benchmark_returns: pd.Series,          
                 total_capital: float = 100.0):

        returns = test_prices[self.best_stock].pct_change().fillna(0)
        equity  = total_capital * (1 + returns).cumprod()

        final_returns = equity.pct_change().fillna(0)
        metrics = _compute_metrics(equity, benchmark_returns, total_capital)
        return equity, final_returns, metrics

# Baseline 5 : Market Portfolio
class MarketPortfolioBaseline:
    """
    True market portfolio: tracks the NIFTY 50 index level directly.
    """

    def __init__(self, index_prices: pd.Series):
        self.index_prices = index_prices

    def fit(self, train_prices: pd.DataFrame) -> None:
        pass

    def backtest(self,
                 test_prices: pd.DataFrame,
                 benchmark_returns: pd.Series,          
                 total_capital: float = 100.0):

        index_prices = self.index_prices.reindex(test_prices.index).ffill().bfill()
        equity       = total_capital * (index_prices / index_prices.iloc[0])

        returns = equity.pct_change().fillna(0)
        metrics = _compute_metrics(equity, benchmark_returns, total_capital)
        return equity, returns, metrics

# Walk-forward engine
def run_walk_forward_baselines(prices: pd.DataFrame,
                               index_prices: pd.Series,
                               total_capital: float = 100.0,
                               risk_free_rate: float = 0.0):

    windows = [
        ("2014", "2016", "2018"),
        ("2016", "2018", "2020"),
        ("2018", "2020", "2022"),
        ("2020", "2022", "2024"),
        ("2022", "2024", "2026"),
    ]

    baselines = {
        "40-60":       FortysixtBaseline(risk_free_rate),
        "EqualWeight": EqualWeightBaseline(),
        "EW_BuyHold":  EqualWeightBuyHoldBaseline(),
        "Greedy":      GreedyBaseline(),
        "Market":      MarketPortfolioBaseline(index_prices),
    }

    nsei_returns_full = index_prices.pct_change().fillna(0)

    eq_segments = {k: [] for k in baselines}
    capital     = {k: total_capital for k in baselines}

    for s, m, e in windows:
        train = prices.loc[s:m]
        test  = prices.loc[m:e].iloc[1:]   # drop the overlap row

        bench_window = nsei_returns_full.reindex(test.index).fillna(0)

        for name, b in baselines.items():
            b.fit(train)
            eq, _, _ = b.backtest(test, bench_window, capital[name])
            eq_segments[name].append(eq)
            capital[name] = eq.iloc[-1]

    benchmark_full = nsei_returns_full.loc["2016":"2026"]

    final_eq = {}
    metrics  = {}

    for name, segs in eq_segments.items():
        eq = pd.concat(segs)
        eq = eq[~eq.index.duplicated(keep="last")]
        final_eq[name] = eq
        metrics[name]  = _compute_metrics(eq, benchmark_full, total_capital)

    return final_eq, metrics


if __name__ == "__main__":

    prices = download_price_data(NIFTY100, start="2014-01-01", end="2026-01-01")
    prices = prices.dropna(axis=1, how="any")

    index_prices = download_price_data(
        ["^NSEI"], start="2014-01-01", end="2026-01-01"
    ).squeeze()

    eq, metrics = run_walk_forward_baselines(
        prices,
        index_prices,
        total_capital=100,
        risk_free_rate=0.065,
    )

    print("\nRESULTS\n")
    for k, v in metrics.items():
        _print_metrics(k, v)

    plt.figure(figsize=(12, 6))
    for k, v in eq.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.title("Equity Curves — Walk-forward baselines (benchmark: ^NSEI)")
    plt.tight_layout()
    plt.show()
