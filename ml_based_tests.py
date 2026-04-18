"""
ml_based_tests.py
=================
Companion to cointegration_based_tests.py.

Runs walk-forward backtests for all ML model variants through the shared
backtests.walk_forward_pipeline.run_walk_forward_backtest() runner.

Grid:
    models : DNN-only, GBT-only, RAF-only, Ensemble (DNN+GBT+RAF)
    k      : 5, 10

All results printed in the same PrettyTable format as cointegration_based_tests.py,
sorted by final capital descending, and plotted via plot_equity_curves().
"""

from config.universe import NIFTY100
from data.downloader import download_price_data
from backtests.walk_forward_pipeline import run_walk_forward_backtest
from cointegration_based_tests import plot_equity_curves

from ml_based.strategy.pipeline import MLPipeline
from prettytable import PrettyTable
from prettytable import TableStyle
import itertools


def run_backtest_ml(prices, total_capital=100.0, k=10, models=("dnn", "gbt", "raf"), min_prob_threshold=0.55):
    """
    Construct an MLPipeline and run it through the shared walk-forward runner.

    Parameters
    ----------
    prices             : pd.DataFrame  -- daily close prices
    total_capital      : float         -- starting capital
    k                  : int           -- long/short leg size
    models             : tuple         -- base learners to include
    min_prob_threshold : float         -- daily confidence gate (0.5 = disabled)

    Returns
    -------
    (equity_series, metrics_dict)
    """
    pipeline = MLPipeline(
        k=k,
        total_capital=total_capital,
        models=models,
        min_prob_threshold=min_prob_threshold,
    )
    return run_walk_forward_backtest(prices, pipeline=pipeline)


if __name__ == "__main__":
    tickers = NIFTY100
    capital = 100.0

    prices = download_price_data(tickers)

    print("\n" + "=" * 70)
    print("ML STATISTICAL ARBITRAGE -- WALK-FORWARD BACKTEST (Nifty 100)")
    print("=" * 70)

    # Grid: 4 model configs x 2 k values = 8 runs
    model_configs = [
        ("XGB",      ("xgb",)),
        ("DNN",      ("dnn",)),
        ("GBT",      ("gbt",)),
        ("RAF",      ("raf",)),
        ("Ensemble", ("dnn", "gbt", "raf", "xgb")),
    ]
    # k_values = [5, 10]
    k_values = [5]  # For quicker testing; switch to [5, 10] for full grid

    equity_dict    = {}
    metrics_mapping = {}
    model_table = PrettyTable()
    model_table.set_style(TableStyle.MARKDOWN) 
    for (model_label, models), k in itertools.product(model_configs, k_values):
        label = f"ML-{model_label}-k{k}"
        print(f"\nRunning: {label}")
        equity, metrics = run_backtest_ml(
            prices,
            total_capital=capital,
            k=k,
            models=models,
            min_prob_threshold=0.0,
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

    print("\nWalk-Forward ML Backtest Completed.")

    # ------------------------------------------------------------------
    # Side-by-side comparison with cointegration strategies
    # Run cointegration_based_tests.py first to populate coint_equity_dict,
    # then uncomment:
    # ------------------------------------------------------------------
    # combined = {**equity_dict, **coint_equity_dict}
    # plot_equity_curves(combined)
