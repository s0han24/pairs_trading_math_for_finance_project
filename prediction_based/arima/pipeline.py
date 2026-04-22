"""
ARIMA-based baseline statistical arbitrage pipeline for Nifty 100.

Design mirrors MLPipeline exactly so it can be dropped into the same
walk_forward_pipeline runner without modification:

    fit(train_prices)            -> fits per-stock ARIMA models, stores warmup prefix
    backtest(test_prices, plot)  -> returns (equity_series, None)

Strategy
--------
For each trading day t:
  1. For every stock s, fit (or retrieve) an ARIMA(p,d,q) model on the price
     series ending at t.
  2. Produce the 1-step-ahead point forecast of the *return* for day t+1.
  3. Rank all stocks by forecast return in descending order.
  4. Long the top-k, short the bottom-k (dollar-neutral, equal weight).
  5. Confidence gate: only trade if mean(top-k forecast) >= min_ret_threshold.

ARIMA order selection
---------------------
Default order is (2, 0, 2) on log-returns -- a reasonable fixed-order
baseline that avoids the computational cost of per-stock AIC search.
Set `auto_order=True` to enable pmdarima's auto_arima (slower but better).

Warmup
------
The last `lookback` days of training prices are stored as a warmup prefix
and prepended to test_prices so that each test-day ARIMA has adequate history
from day 1 (no cold-start dead zone).  Default lookback = 60 trading days.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_ORDER    = (2, 0, 2)   # ARIMA(p, d, q) applied to log-returns
_DEFAULT_LOOKBACK = 252         # matches MLPipeline's 1-year lookback
_MIN_SERIES_LEN   = 20          # minimum observations needed to fit ARIMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_returns(prices: pd.Series) -> pd.Series:
    """Compute log-returns, drop the leading NaN."""
    return np.log(prices / prices.shift(1)).dropna()


def _fit_arima_and_forecast(series: pd.Series, order: tuple) -> float:
    """
    Fit ARIMA(order) on `series` (log-returns) and return the 1-step forecast.
    Returns np.nan on any failure.
    """
    if len(series) < _MIN_SERIES_LEN:
        return np.nan
    try:
        model  = ARIMA(series.values, order=order, trend="n")
        result = model.fit()
        fc     = result.forecast(steps=1)
        return float(fc[0])
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class ARIMAPipeline:
    """
    ARIMA-based baseline that mirrors MLPipeline's external interface.

    Parameters
    ----------
    k                 : long + short leg size (default 10).
    total_capital     : starting / carried-forward capital.
    order             : ARIMA (p, d, q) applied to log-returns.
                        Default (2, 0, 2).
    lookback          : number of days used for each ARIMA fit window.
                        Default 60.  Larger = more stable; slower.
    min_ret_threshold : minimum mean forecast return of top-k required to trade.
                        Mirrors MLPipeline's confidence gate.  Set to -inf to
                        disable.  Default 0.0 (any positive expected return).
    refit_every       : refit ARIMA models every N test days.  1 = daily refit
                        (most accurate, slowest).  Default 1.
    """

    def __init__(
        self,
        k=10,
        total_capital=100.0,
        order=_DEFAULT_ORDER,
        lookback=_DEFAULT_LOOKBACK,
        min_ret_threshold=0.0,
        refit_every=1,
    ):
        self.k                 = k
        self.total_capital     = total_capital
        self.order             = order
        self.lookback          = lookback
        self.min_ret_threshold = min_ret_threshold
        self.refit_every       = refit_every

        self._fitted        = False
        self._warmup_prices = None   # last `lookback` rows of training data

    # ------------------------------------------------------------------
    def fit(self, train_prices: pd.DataFrame):
        """
        Store the warmup prefix (last `lookback` days of training prices).
        ARIMA models are fitted on-the-fly during backtest to always use the
        most recent `lookback`-day window -- this is the standard approach for
        rolling-window time-series forecasting and avoids stale in-sample fits.

        Parameters
        ----------
        train_prices : pd.DataFrame  shape (T_train, n_stocks), daily prices
        """
        n_rows = len(train_prices)
        prefix_len = min(self.lookback, n_rows)

        self._warmup_prices = train_prices.iloc[-prefix_len:].copy()
        self._fitted        = True

        print(
            f"  ARIMAPipeline.fit() -- warmup prefix: {prefix_len} days, "
            f"order: {self.order}, lookback: {self.lookback}"
        )
        return self

    # ------------------------------------------------------------------
    def _forecast_all(
        self, prices_window: pd.DataFrame
    ) -> pd.Series:
        """
        For each stock in `prices_window`, fit ARIMA on its log-return series
        and return a Series of 1-step-ahead log-return forecasts.

        Parameters
        ----------
        prices_window : pd.DataFrame  shape (lookback+1, n_stocks)
                        Must have at least _MIN_SERIES_LEN rows.

        Returns
        -------
        forecasts : pd.Series  index=ticker, value=forecast log-return
        """
        forecasts = {}
        for ticker in prices_window.columns:
            col = prices_window[ticker].dropna()
            if len(col) < _MIN_SERIES_LEN + 1:
                continue
            lr = _log_returns(col)
            fc = _fit_arima_and_forecast(lr, self.order)
            if not np.isnan(fc):
                forecasts[ticker] = fc

        return pd.Series(forecasts, name="forecast_return")

    # ------------------------------------------------------------------
    def backtest(
        self, test_prices: pd.DataFrame, plot: bool = False
    ):
        """
        Walk-forward backtest over test_prices.

        Steps each day:
          1. Build price window = warmup prefix + test prices up to (and
             including) the current date.  Truncated to last `lookback` days.
          2. Forecast 1-step log-returns for all stocks via rolling ARIMA.
          3. Rank by forecast descending; long top-k, short bottom-k.
          4. Confidence gate: skip day if mean(top-k forecast) < min_ret_threshold.
          5. Realise P&L from next-day simple returns; compound capital.

        Returns
        -------
        (equity_series, None)  -- None keeps the interface consistent with
                                  CointegrationPipeline / MLPipeline.
        """
        if not self._fitted:
            print("  ARIMAPipeline not fitted -- holding cash.")
            return (
                pd.Series(self.total_capital, index=test_prices.index),
                None,
            )

        k       = self.k
        capital = self.total_capital

        # ------------------------------------------------------------------ #
        # Build the combined price history: warmup prefix + test window.
        # We keep only columns present in both to avoid NaN bleed.
        # ------------------------------------------------------------------ #
        if self._warmup_prices is not None:
            shared_cols        = test_prices.columns.intersection(
                self._warmup_prices.columns
            )
            prices_full = pd.concat(
                [self._warmup_prices[shared_cols], test_prices[shared_cols]]
            )
            prices_full = prices_full[~prices_full.index.duplicated(keep="last")]
        else:
            prices_full = test_prices.copy()

        test_dates       = test_prices.index
        next_day_returns = test_prices.pct_change(1)  # simple returns for P&L

        equity       = [capital]
        equity_dates = [test_dates[0]]
        days_traded  = 0
        days_cash    = 0

        for i, date in enumerate(test_dates[:-1]):
            next_date = test_dates[i + 1]

            # ---------------------------------------------------------------- #
            # Refit cadence -- skip forecasting on non-refit days and carry
            # forward last ranking.  Day 0 always forecasts.
            # ---------------------------------------------------------------- #
            if i % self.refit_every == 0:
                # Slice: all history up to and including `date`, last `lookback` rows
                loc    = prices_full.index.get_loc(date)
                start  = max(0, loc - self.lookback + 1)
                window = prices_full.iloc[start: loc + 1]

                forecasts = self._forecast_all(window)
                ranking   = forecasts.sort_values(ascending=False)

            # ---------------------------------------------------------------- #
            # Need at least 2k stocks with valid forecasts
            # ---------------------------------------------------------------- #
            if len(ranking) < 2 * k:
                equity.append(equity[-1])
                equity_dates.append(next_date)
                days_cash += 1
                continue

            long_stocks  = ranking.index[:k].tolist()
            short_stocks = ranking.index[-k:].tolist()

            # Confidence gate
            if ranking.iloc[:k].mean() < self.min_ret_threshold:
                equity.append(equity[-1])
                equity_dates.append(next_date)
                days_cash += 1
                continue

            # Dollar-neutral equal-weight P&L
            pos_size  = equity[-1] / (2 * k)
            day_ret   = next_day_returns.loc[next_date]

            long_pnl  = sum(pos_size * day_ret.get(t, 0.0) for t in long_stocks)
            short_pnl = sum(-pos_size * day_ret.get(t, 0.0) for t in short_stocks)

            new_capital = equity[-1] + long_pnl + short_pnl
            equity.append(max(new_capital, 0.0))
            equity_dates.append(next_date)
            days_traded += 1

        total_days = days_traded + days_cash
        if total_days > 0:
            print(
                f"  Traded {days_traded}/{total_days} days "
                f"({days_traded / total_days:.1%} active, "
                f"threshold={self.min_ret_threshold}, k={k}, "
                f"order={self.order}, lookback={self.lookback})"
            )

        return pd.Series(equity, index=equity_dates), None


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yfinance as yf

    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
               "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS",
               "BAJFINANCE.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
               "ASIANPAINT.NS", "MARUTI.NS", "ULTRACEMCO.NS",
               "WIPRO.NS", "NESTLEIND.NS", "TATAMOTORS.NS",
               "SUNPHARMA.NS", "TITAN.NS"]

    print("Downloading price data ...")
    raw = yf.download(tickers, start="2019-01-01", end="2022-12-31",
                      auto_adjust=True, progress=False)["Close"]
    raw.dropna(axis=1, how="any", inplace=True)
    prices = raw.copy()

    split        = int(len(prices) * 0.7)
    train_prices = prices.iloc[:split]
    test_prices  = prices.iloc[split:]

    pipeline = ARIMAPipeline(k=5, total_capital=100.0, lookback=30,
                             min_ret_threshold=0.0, refit_every=1)
    pipeline.fit(train_prices)

    equity, _ = pipeline.backtest(test_prices)

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    print(f"\nFinal equity : {equity.iloc[-1]:.4f}")
    print(f"Total return : {total_return:.2f}%")
    print(equity.tail())
