"""
ml_based_tests.py
=================
Companion to cointegration_based_tests.py.

Runs walk-forward backtests for all four ML model variants through the
**shared** backtests.walk_forward_pipeline.run_walk_forward_backtest() runner,
the same runner used by the cointegration strategies.

Strategies evaluated:
    ML-DNN       — deep neural network only
    ML-GBT       — gradient-boosted trees only
    ML-RAF       — random forest only
    ML-Ensemble  — equal-weight average of DNN + GBT + RAF  (paper's primary result)

All results are printed in the same PrettyTable format as cointegration_based_tests.py,
sorted by final capital descending, and plotted via plot_equity_curves().

To compare ML vs cointegration side-by-side, merge the two equity_dicts and call
plot_equity_curves() on the combined dict (see commented block at the bottom).
"""

from config.universe import NIFTY100
from data.downloader import download_price_data
from backtests.walk_forward_pipeline import run_walk_forward_backtest
from cointegration_based_tests import plot_equity_curves

from ml_based.strategy.pipeline import MLPipeline
from prettytable import PrettyTable

import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)


def run_backtest_ml(prices, total_capital=100.0, k=10, models=("dnn", "gbt", "raf")):
    """
    Thin wrapper: constructs an MLPipeline and hands it to the shared runner.

    Parameters
    ----------
    prices        : pd.DataFrame -- daily close prices
    total_capital : float        -- starting capital
    k             : int          -- number of long / short positions per day
    models        : tuple        -- subset of ("dnn", "gbt", "raf") to include

    Returns
    -------
    (equity_series, metrics_dict)
    """
    pipeline = MLPipeline(k=k, total_capital=total_capital, models=models)
    return run_walk_forward_backtest(prices, pipeline=pipeline)


if __name__ == "__main__":
    tickers = NIFTY100
    k = 10          # paper default; set to 5 to match cointegration pipeline's k=5
    capital = 100.0

    prices = download_price_data(tickers)

    print("\n" + "=" * 70)
    print("ML STATISTICAL ARBITRAGE -- WALK-FORWARD BACKTEST (Nifty 100)")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Run all four model variants
    # ------------------------------------------------------------------ #
    configs = [
        ("ML-DNN",      ("dnn",)),
        ("ML-GBT",      ("gbt",)),
        ("ML-RAF",      ("raf",)),
        ("ML-Ensemble", ("dnn", "gbt", "raf")),
    ]

    equity_dict    = {}
    metrics_mapping = {}

    for label, models in configs:
        print(f"\nRunning backtest for: {label}")
        equity, metrics = run_backtest_ml(
            prices, total_capital=capital, k=k, models=models
        )
        equity_dict[label]     = equity
        metrics_mapping[label] = metrics

    # ------------------------------------------------------------------ #
    # Print results table -- sorted by final capital descending
    # ------------------------------------------------------------------ #
    table = PrettyTable()
    table.field_names = [
        "Strategy", "Final Capital", "Max Drawdown",
        "Sharpe Ratio", "Alpha", "Beta", "Annual Volatility",
    ]

    sorted_labels = sorted(
        equity_dict.keys(),
        key=lambda lbl: metrics_mapping[lbl]["final_capital"],
        reverse=True,
    )

    for label in sorted_labels:
        m = metrics_mapping[label]
        table.add_row([
            label,
            f"{m['final_capital']:.2f}",
            f"{m['max_drawdown']:.2%}",
            f"{m['sharpe_ratio']:.2f}",
            f"{m['alpha']:.4f}",
            f"{m['beta']:.4f}",
            f"{m['annual_volatility']:.2%}",
        ])

    print("\n")
    print(table)

    # ------------------------------------------------------------------ #
    # Equity curves
    # ------------------------------------------------------------------ #
    plot_equity_curves(equity_dict)

    print("\nWalk-Forward ML Backtest Completed.")

    # ------------------------------------------------------------------ #
    # Side-by-side comparison with cointegration strategies
    # Run cointegration_based_tests.py first, then uncomment:
    # ------------------------------------------------------------------ #
    # combined = {**equity_dict, **coint_equity_dict}
    # plot_equity_curves(combined)
