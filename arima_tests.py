"""
arima_based_tests.py
====================
Companion to ml_based_tests.py.

Runs walk-forward backtests for all ARIMA model variants through the shared
backtests.walk_forward_pipeline.run_walk_forward_backtest() runner.

Grid:
    order    : (1,0,0), (2,0,2), (1,0,1), (5,0,0)
    k        : 5, 10

All results printed in the same PrettyTable format as ml_based_tests.py,
sorted by final capital descending, and plotted via plot_equity_curves().
"""

from config.universe import NIFTY100
from data.downloader import download_price_data
from backtests.walk_forward_pipeline import run_walk_forward_backtest
from cointegration_based_tests import plot_equity_curves

from prediction_based.arima.pipeline import ARIMAPipeline
from prettytable import PrettyTable
from prettytable import TableStyle
import itertools

import numpy as np
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


def run_backtest_arima(prices, total_capital=100.0, k=10, order=(2, 0, 2),
                       lookback=60, min_ret_threshold=0.0, refit_every=1):
    """
    Construct an ARIMAPipeline and run it through the shared walk-forward runner.

    Parameters
    ----------
    prices            : pd.DataFrame  -- daily close prices
    total_capital     : float         -- starting capital
    k                 : int           -- long/short leg size
    order             : tuple         -- ARIMA (p, d, q) order on log-returns
    lookback          : int           -- rolling window length for each ARIMA fit
    min_ret_threshold : float         -- daily confidence gate (0.0 = any positive forecast)
    refit_every       : int           -- refit models every N days (1 = daily)

    Returns
    -------
    (equity_series, metrics_dict)
    """
    pipeline = ARIMAPipeline(
        k=k,
        total_capital=total_capital,
        order=order,
        lookback=lookback,
        min_ret_threshold=min_ret_threshold,
        refit_every=refit_every,
    )
    return run_walk_forward_backtest(prices, pipeline=pipeline)


if __name__ == "__main__":
    tickers = NIFTY100
    capital = 100.0

    prices = download_price_data(tickers)

    print("\n" + "=" * 70)
    print("ARIMA STATISTICAL ARBITRAGE -- WALK-FORWARD BACKTEST (Nifty 100)")
    print("=" * 70)

    # Grid: 4 ARIMA orders x 1 k value = 4 runs
    # Each label maps to an ARIMA (p, d, q) order applied to log-returns.
    order_configs = [
        ("AR1",       (1, 0, 0)),   # pure AR(1) -- simplest possible baseline
        ("ARIMA202",  (2, 0, 2)),   # balanced ARMA -- default recommended order
        ("ARIMA101",  (1, 0, 1)),   # lightweight ARMA
        ("AR5",       (5, 0, 0)),   # longer AR memory
    ]
    # k_values = [5, 10]
    k_values = [5]  # For quicker testing; switch to [5, 10] for full grid

    equity_dict     = {}
    metrics_mapping = {}
    model_table = PrettyTable()
    model_table.set_style(TableStyle.MARKDOWN)

    for (order_label, order), k in itertools.product(order_configs, k_values):
        label = f"ARIMA-{order_label}-k{k}"
        print(f"\nRunning: {label}")
        equity, metrics = run_backtest_arima(
            prices,
            total_capital=capital,
            k=k,
            order=order,
            lookback=60,  # 2-3 months of trading days for each ARIMA fit
            min_ret_threshold=0.0,
            refit_every=5,
        )
        equity_dict[label]     = equity
        metrics_mapping[label] = metrics

        model_table.field_names = ["Metric", "Value"]
        for m_key in ['final_capital', 'max_drawdown', 'sharpe_ratio', 'alpha', 'beta', 'annual_volatility']:
            m_value = metrics[m_key]
            if isinstance(m_value, float):
                if "drawdown" in m_key or "volatility" in m_key:
                    formatted_value = f"{m_value:.2%}"
                elif "capital" in m_key:
                    formatted_value = f"${m_value:.2f}"
                else:
                    formatted_value = f"{m_value:.4f}"
            else:
                formatted_value = str(m_value)
            model_table.add_row([m_key, formatted_value])
        print(model_table)
        model_table.clear_rows()  # Clear rows for next model's metrics

    # ------------------------------------------------------------------
    # Results table sorted by final capital descending
    # ------------------------------------------------------------------
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

    plot_equity_curves(equity_dict)

    print("\nWalk-Forward ARIMA Backtest Completed.")

    # ------------------------------------------------------------------
    # Side-by-side comparison with ML strategies
    # Run ml_based_tests.py first to populate ml_equity_dict,
    # then uncomment:
    # ------------------------------------------------------------------
    # combined = {**equity_dict, **ml_equity_dict}
    # plot_equity_curves(combined)
