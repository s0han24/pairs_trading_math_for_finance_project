from .metrics import sharpe_ratio, max_drawdown, compute_alpha_beta, compute_annual_volatility
import pandas as pd

def run_walk_forward_backtest(prices, pipeline):
    
    # # 2 years (approx 504 trading days) in sample, 2 years out sample
    # # Walk forward windows:
    # # 1: 2014-2016 in, 2016-2018 out
    # # 2: 2016-2018 in, 2018-2020 out
    # # 3: 2018-2020 in, 2020-2022 out
    # # 4: 2020-2022 in, 2022-2024 out
    # # 5: 2022-2024 in, 2024-2026 out
    
    # windows = [
    #     ("2014-01-01", "2016-01-01", "2018-01-01"),
    #     ("2016-01-01", "2018-01-01", "2020-01-01"),
    #     ("2018-01-01", "2020-01-01", "2022-01-01"),
    #     ("2020-01-01", "2022-01-01", "2024-01-01"),
    #     ("2022-01-01", "2024-01-01", "2026-01-01")
    # ]
    
    # 3 years (approx 756 trading days) in sample, 1 year out sample
    # Walk forward windows:
    windows = [
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
    
    all_equity = []
    benchmark_returns_list = []
    
    sub_period_metrics = []

    initial_capital = pipeline.total_capital
    current_capital = initial_capital
    
    for start_in, end_in_start_ood, end_ood in windows:
        print(f"\n--- Walk Forward Step: Train [{start_in} to {end_in_start_ood}], Test [{end_in_start_ood} to {end_ood}] ---")
        
        train_prices = prices.loc[start_in:end_in_start_ood]
        test_prices = prices.loc[end_in_start_ood:end_ood]
        
        # Fit on training data
        fit_result = pipeline.fit(train_prices)
        
        # Check if pipeline is pairs-based
        if hasattr(pipeline, 'selected_pairs'):
            selected_pairs = pipeline.selected_pairs
            print(f"Selected {len(selected_pairs)} pairs.")
            for s1, s2, corr, pvalue in selected_pairs:
                print(f"  {s1}-{s2} (P-value: {pvalue:.4f}, Corr: {corr:.2f})")
            
            if not selected_pairs:
                print("No pairs found. Holding cash.")
                # Equity remains flat
                cash_eq = pd.Series(current_capital, index=test_prices.index)
                all_equity.append(cash_eq)
                continue
        else:
            print("Model fitted.")
            
        # Benchmark
        bench_ret = test_prices.pct_change().mean(axis=1).fillna(0)
        benchmark_returns_list.append(bench_ret)
        
        # Backtest on test data
        pipeline.total_capital = current_capital
        equity_series, _ = pipeline.backtest(test_prices, plot=False)
        all_equity.append(equity_series)
        current_capital = equity_series.iloc[-1]
        
        # Calculate sub-period metrics
        period_returns = equity_series.pct_change().dropna()
        period_bench_ret = bench_ret.reindex(period_returns.index).fillna(0)
        
        sub_metrics = {
            'period': f"{end_in_start_ood} to {end_ood}",
            'final_capital': current_capital,
            'max_drawdown': max_drawdown(equity_series),
            'sharpe_ratio': sharpe_ratio(period_returns),
            'annual_volatility': compute_annual_volatility(period_returns),
            'alpha': compute_alpha_beta(period_returns, period_bench_ret)[0],
            'beta': compute_alpha_beta(period_returns, period_bench_ret)[1]
        }
        sub_period_metrics.append(sub_metrics)
            
    # Combine walk-forward pieces 
    # Drop duplicates at boundaries
    total_portfolio_equity = pd.concat(all_equity)
    total_portfolio_equity = total_portfolio_equity[~total_portfolio_equity.index.duplicated(keep='last')]
    
    total_benchmark_ret = pd.concat(benchmark_returns_list)
    total_benchmark_ret = total_benchmark_ret[~total_benchmark_ret.index.duplicated(keep='last')]
    
    # Compute overall metrics
    returns = total_portfolio_equity.pct_change().dropna()
    bench_returns = total_benchmark_ret.reindex(returns.index).fillna(0)
    
    mdd = max_drawdown(total_portfolio_equity)
    sr = sharpe_ratio(returns)
    alpha, beta = compute_alpha_beta(returns, bench_returns)
    vol = compute_annual_volatility(returns)
    
    metrics_dict = {
        'final_capital': total_portfolio_equity.iloc[-1],
        'max_drawdown': mdd,
        'sharpe_ratio': sr,
        'alpha': alpha,
        'beta': beta,
        'annual_volatility': vol,
        'sub_periods': sub_period_metrics
    }
    
    print("\n--- Sub-Period Performance Summary ---")
    for cp in sub_period_metrics:
        print(f"Period: {cp['period']} | Final Capital: ${cp['final_capital']:.2f} | Sharpe: {cp['sharpe_ratio']:.2f} | MDD: {cp['max_drawdown']:.2%} | Vol: {cp['annual_volatility']:.2%}")
        
    return total_portfolio_equity, metrics_dict
