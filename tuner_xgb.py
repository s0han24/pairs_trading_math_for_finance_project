"""
tuner_xgb.py
============
Optuna hyperparameter tuner for the XGBoost-only MLPipeline variant.

Tuneable parameters
-------------------
Pipeline-level:
    k                  – long + short leg size  (3 … 15)
    min_prob_threshold – confidence gate before trading  (0.50 … 0.70)

XGBoost model-level:
    xgb_n_estimators       – number of boosting rounds  (50 … 500)
    xgb_max_depth          – tree depth  (2 … 8)
    xgb_learning_rate      – step size shrinkage  (0.01 … 0.30)
    xgb_subsample          – row subsampling ratio  (0.5 … 1.0)
    xgb_colsample_bytree   – feature subsampling per tree  (0.5 … 1.0)
    xgb_gamma              – min loss reduction for a split  (0 … 5)
    xgb_min_child_weight   – min sum of instance weight in a child  (1 … 10)
    xgb_reg_alpha          – L1 regularisation  (0 … 1)
    xgb_reg_lambda         – L2 regularisation  (0.5 … 5)

Objective: maximise the walk-forward Sharpe Ratio over ALL folds
           (same 9-fold expanding-window schedule as the main backtests).

Usage
-----
    python tuner_xgb.py                    # 100 trials, results to stdout + CSV
    python tuner_xgb.py --n-trials 50      # quick smoke-test
    python tuner_xgb.py --n-trials 100 --study-name my_study  # named study
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import random

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config.universe import NIFTY100
from data.downloader import download_price_data
from backtests.walk_forward_pipeline import run_walk_forward_backtest
from ml_based.strategy.pipeline import MLPipeline

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# Walk-forward windows (identical to walk_forward_pipeline.py)
# ---------------------------------------------------------------------------
WINDOWS = [
    ("2014-01-01", "2017-01-01", "2018-01-01"),
    ("2015-01-01", "2018-01-01", "2019-01-01"),
    ("2016-01-01", "2019-01-01", "2020-01-01"),
    ("2017-01-01", "2020-01-01", "2021-01-01"),
    ("2018-01-01", "2021-01-01", "2022-01-01"),
    ("2019-01-01", "2022-01-01", "2023-01-01"),
    ("2020-01-01", "2023-01-01", "2024-01-01"),
    ("2021-01-01", "2024-01-01", "2025-01-01"),
    ("2022-01-01", "2025-01-01", "2026-01-01"),
]


# ---------------------------------------------------------------------------
# Helper: run a single trial configuration
# ---------------------------------------------------------------------------

def _run_trial_config(prices: pd.DataFrame, params: dict) -> dict:
    """
    Build an XGB-only MLPipeline from `params`, run the full walk-forward
    backtest, and return the aggregate metrics dict.
    """
    pipeline = MLPipeline(
        k=params["k"],
        total_capital=100.0,
        models=("xgb",),
        min_prob_threshold=params["min_prob_threshold"],
        xgb_n_estimators=params["xgb_n_estimators"],
        xgb_max_depth=params["xgb_max_depth"],
        xgb_learning_rate=params["xgb_learning_rate"],
        xgb_subsample=params["xgb_subsample"],
        xgb_colsample_bytree=params["xgb_colsample_bytree"],
        xgb_gamma=params["xgb_gamma"],
        xgb_min_child_weight=params["xgb_min_child_weight"],
        xgb_reg_alpha=params["xgb_reg_alpha"],
        xgb_reg_lambda=params["xgb_reg_lambda"],
    )
    _, metrics = run_walk_forward_backtest(prices, pipeline=pipeline)
    return metrics


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(prices: pd.DataFrame):
    """
    Returns an Optuna objective function closed over the price data.

    Objective value: walk-forward Sharpe Ratio (maximise).
    Secondary values stored as trial user_attrs for later analysis.
    """

    def objective(trial: optuna.Trial) -> float:
        # ------------------------------------------------------------------ #
        # 1. Sample hyperparameters
        # ------------------------------------------------------------------ #
        params = {
            # --- Pipeline-level ---
            "k": trial.suggest_int("k", 3, 15),
            "min_prob_threshold": trial.suggest_float(
                "min_prob_threshold", 0.50, 0.70, step=0.01
            ),
            # --- XGBoost model ---
            "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 50, 500, step=50),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 2, 8),
            "xgb_learning_rate": trial.suggest_float(
                "xgb_learning_rate", 0.01, 0.30, log=True
            ),
            "xgb_subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0, step=0.05),
            "xgb_colsample_bytree": trial.suggest_float(
                "xgb_colsample_bytree", 0.5, 1.0, step=0.05
            ),
            "xgb_gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
            "xgb_min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
            "xgb_reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 1.0),
            "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.5, 5.0),
        }

        print(f"\n[Trial {trial.number}] Params: {params}")
        t0 = time.time()

        # ------------------------------------------------------------------ #
        # 2. Run full walk-forward evaluation
        # ------------------------------------------------------------------ #
        try:
            metrics = _run_trial_config(prices, params)
        except Exception as exc:
            print(f"  [Trial {trial.number}] FAILED: {exc}")
            return float("-inf")

        elapsed = time.time() - t0

        # ------------------------------------------------------------------ #
        # 3. Store secondary metrics as trial attributes
        # ------------------------------------------------------------------ #
        trial.set_user_attr("final_capital",    round(metrics["final_capital"], 4))
        trial.set_user_attr("max_drawdown",     round(metrics["max_drawdown"], 6))
        trial.set_user_attr("sharpe_ratio",     round(metrics["sharpe_ratio"], 6))
        trial.set_user_attr("alpha",            round(metrics["alpha"], 6))
        trial.set_user_attr("beta",             round(metrics["beta"], 6))
        trial.set_user_attr("annual_volatility",round(metrics["annual_volatility"], 6))
        trial.set_user_attr("elapsed_sec",      round(elapsed, 1))

        # Per-fold Sharpe (stored as JSON string)
        fold_sharpes = [
            round(sp["sharpe_ratio"], 4)
            for sp in metrics.get("sub_periods", [])
        ]
        trial.set_user_attr("fold_sharpes", json.dumps(fold_sharpes))

        sr = metrics["sharpe_ratio"]
        print(
            f"  [Trial {trial.number}] Sharpe={sr:.4f} | "
            f"Capital=${metrics['final_capital']:.2f} | "
            f"MDD={metrics['max_drawdown']:.2%} | "
            f"Alpha={metrics['alpha']:.4f} | "
            f"Elapsed={elapsed:.0f}s"
        )
        return sr  # maximise

    return objective


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------

def _print_top_trials(study: optuna.Study, n: int = 10) -> None:
    """Print the top-n trials sorted by Sharpe Ratio."""
    trials = [t for t in study.trials if t.value is not None and t.value > float("-inf")]
    trials.sort(key=lambda t: t.value, reverse=True)

    print("\n" + "=" * 80)
    print(f"  TOP {min(n, len(trials))} TRIALS  (by Sharpe Ratio)")
    print("=" * 80)

    header = (
        f"{'Rank':>4}  {'Trial':>6}  {'Sharpe':>7}  {'Capital':>10}  "
        f"{'MDD':>7}  {'Alpha':>7}  k  thresh  "
        f"ne   d   lr      ss    cb    g     mcw  a      l"
    )
    print(header)
    print("-" * len(header))

    for rank, t in enumerate(trials[:n], start=1):
        p = t.params
        ua = t.user_attrs
        print(
            f"{rank:>4}  {t.number:>6}  {t.value:>7.4f}  "
            f"${ua.get('final_capital', 0):>9.2f}  "
            f"{ua.get('max_drawdown', 0):>7.2%}  "
            f"{ua.get('alpha', 0):>7.4f}  "
            f"{p['k']:>2}  {p['min_prob_threshold']:>5.2f}  "
            f"{p['xgb_n_estimators']:>4}  {p['xgb_max_depth']:>2}  "
            f"{p['xgb_learning_rate']:>6.4f}  {p['xgb_subsample']:>4.2f}  "
            f"{p['xgb_colsample_bytree']:>4.2f}  {p['xgb_gamma']:>5.2f}  "
            f"{p['xgb_min_child_weight']:>4}  "
            f"{p['xgb_reg_alpha']:>5.3f}  {p['xgb_reg_lambda']:>5.3f}"
        )

    # Best trial details
    best = study.best_trial
    print("\n" + "=" * 80)
    print("  BEST TRIAL DETAIL")
    print("=" * 80)
    print(f"  Trial number   : {best.number}")
    print(f"  Sharpe Ratio   : {best.value:.6f}")
    for k, v in best.user_attrs.items():
        print(f"  {k:<22}: {v}")
    print("\n  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<28}: {v}")

    # Per-fold breakdown for best trial
    fold_sharpes = json.loads(best.user_attrs.get("fold_sharpes", "[]"))
    if fold_sharpes:
        print("\n  Per-fold Sharpe Ratios:")
        for i, (window, fs) in enumerate(zip(WINDOWS, fold_sharpes), start=1):
            period = f"{window[1][:4]}–{window[2][:4]}"
            bar = "█" * max(0, int(fs * 8)) if fs > 0 else "▒" * 3
            print(f"    Fold {i} ({period}): {fs:>7.4f}  {bar}")


def _save_results_csv(study: optuna.Study, out_path: Path) -> None:
    """Persist all completed trials to a CSV for external analysis."""
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        row = {"trial": t.number, "sharpe_ratio": t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
    df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="XGBoost MLPipeline Optuna tuner")
    parser.add_argument("--n-trials",   type=int,  default=100,
                        help="Number of Optuna trials (default: 100)")
    parser.add_argument("--study-name", type=str,  default="xgb_pairs_tuner",
                        help="Optuna study name (default: xgb_pairs_tuner)")
    parser.add_argument("--out-csv",    type=str,  default="tuner_xgb_results.csv",
                        help="Path for output CSV (default: tuner_xgb_results.csv)")
    parser.add_argument("--n-jobs",     type=int,  default=1,
                        help="Parallel Optuna workers (default: 1; set >1 with caution)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Download data once (shared across all trials)
    # ------------------------------------------------------------------ #
    print("Downloading price data ...")
    prices = download_price_data(NIFTY100)
    print(f"  Universe: {prices.shape[1]} tickers | {len(prices)} trading days\n")

    # ------------------------------------------------------------------ #
    # 2. Create / load study  (TPE sampler, MedianPruner disabled — we
    #    evaluate ALL folds per trial as requested)
    # ------------------------------------------------------------------ #
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
    )

    # Seed with the current default config so Optuna builds a known baseline
    study.enqueue_trial({
        "k": 5,
        "min_prob_threshold": 0.50,
        "xgb_n_estimators": 100,
        "xgb_max_depth": 3,
        "xgb_learning_rate": 0.10,
        "xgb_subsample": 1.0,
        "xgb_colsample_bytree": 1.0,
        "xgb_gamma": 0.0,
        "xgb_min_child_weight": 1,
        "xgb_reg_alpha": 0.0,
        "xgb_reg_lambda": 1.0,
    })

    # ------------------------------------------------------------------ #
    # 3. Optimise
    # ------------------------------------------------------------------ #
    print(f"Starting optimisation: {args.n_trials} trials, study='{args.study_name}'")
    print("Objective: maximise walk-forward Sharpe Ratio (all 9 folds)\n")
    print("=" * 80)

    study.optimize(
        make_objective(prices),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=False,   # we print trial-level output manually
    )

    # ------------------------------------------------------------------ #
    # 4. Report
    # ------------------------------------------------------------------ #
    _print_top_trials(study, n=10)
    _save_results_csv(study, Path(args.out_csv))

    print("\nOptimisation complete.")


if __name__ == "__main__":
    main()
