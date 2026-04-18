"""
ML-based statistical arbitrage pipeline for Nifty 100.

Faithful implementation of Krauss, Do & Huck (2016):
  - Features : R(m) for m in {1..20, 40, 60, ..., 240}  -> 31 lagged-return features
  - Label    : 1 if stock beats cross-sectional median next-day return, else 0
  - Models   : DNN (MLP), GBT, RAF, equal-weight ensemble of predicted probabilities
  - Trading  : rank all stocks by P(outperform), long top-k, short bottom-k

Key fixes vs. naive version:
  1. Warmup prefix  -- last 240 days of train_prices stored during fit() and
     prepended to test_prices in backtest(), so prediction works from day 1
     of every test window (no dead cash-holding zone).
  2. Confidence gate -- only trade on days where mean(top-k probs) >= min_prob_threshold.
     Censors the uncertain middle of the ranking (paper intent).  Default 0.55.
  3. Configurable k -- default 10 (paper); k=5 recommended for Nifty 100 universe.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature configuration  (paper Section 4.2)
# ---------------------------------------------------------------------------
_LAGS    = list(range(1, 21)) + list(range(40, 241, 20))   # 31 features
_MAX_LAG = max(_LAGS)                                        # 240


# ---------------------------------------------------------------------------
# Training feature builder
# ---------------------------------------------------------------------------

def _compute_features(prices):
    """
    Build (X, y) from a price DataFrame for model training.

    For every stock s and every date t where all 31 lags are available:
        X[t, s] = [R(1), ..., R(20), R(40), ..., R(240)]
        y[t, s] = 1 if R(1)_{t+1} > cross-sectional median R(1)_{t+1}, else 0

    Returns
    -------
    X : pd.DataFrame  MultiIndex(date, ticker), 31 columns
    y : pd.Series     same index, binary label
    """
    ret_panel = {m: prices.pct_change(m) for m in _LAGS}
    next_ret  = prices.pct_change(1).shift(-1)

    valid_dates = prices.index[_MAX_LAG:]
    valid_dates = valid_dates[valid_dates.isin(next_ret.dropna(how="all").index)]

    all_X, all_y, all_idx = [], [], []

    for date in valid_dates:
        nr = next_ret.loc[date].dropna()
        if nr.empty:
            continue
        median_ret = nr.median()

        for ticker in nr.index:
            vals = np.array([ret_panel[m].loc[date, ticker] for m in _LAGS], dtype=float)
            if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
                continue
            all_X.append(vals)
            all_y.append(1 if nr[ticker] > median_ret else 0)
            all_idx.append((date, ticker))

    if not all_X:
        return pd.DataFrame(), pd.Series(dtype=int)

    idx = pd.MultiIndex.from_tuples(all_idx, names=["date", "ticker"])
    X   = pd.DataFrame(all_X,  index=idx, columns=[f"R{m}" for m in _LAGS])
    y   = pd.Series(all_y, index=idx, name="label")
    return X, y


# ---------------------------------------------------------------------------
# Prediction feature builder  (single date, vectorised across stocks)
# ---------------------------------------------------------------------------

def _build_predict_matrix(prices, date):
    """
    Build the 31-feature row for every stock at `date`.

    `prices` must include the warmup prefix so that the index position of
    `date` is >= _MAX_LAG from the start of the DataFrame.

    Returns
    -------
    X_pred  : pd.DataFrame  shape (n_valid_stocks, 31)
    tickers : list[str]     same order as rows
    """
    loc = prices.index.get_loc(date)
    if loc < _MAX_LAG:
        return pd.DataFrame(), []

    window   = prices.iloc[loc - _MAX_LAG: loc + 1]   # shape (_MAX_LAG+1, n_stocks)
    last_row = window.iloc[-1]

    feat_rows = {}
    for m in _LAGS:
        base_row = window.iloc[-1 - m]
        with np.errstate(divide="ignore", invalid="ignore"):
            feat_rows[f"R{m}"] = last_row / base_row - 1.0

    feat_df    = pd.DataFrame(feat_rows)
    valid_mask = feat_df.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    feat_df    = feat_df[valid_mask]

    if feat_df.empty:
        return pd.DataFrame(), []

    return feat_df, feat_df.index.tolist()


# ---------------------------------------------------------------------------
# Model constructors  (paper Section 4.3)
# ---------------------------------------------------------------------------

def _make_dnn():
    """
    Architecture 31-31-10-5-2 (paper topology).
      relu   ~= maxout activation (maxout unavailable in sklearn)
      adam   ~= ADADELTA (both adaptive momentum-based)
      alpha    = 1e-5 matches paper lambda_DNN = 0.00001
      400 epochs as per paper
    """
    return MLPClassifier(
        hidden_layer_sizes=(31, 10, 5),
        activation="relu",
        solver="adam",
        alpha=1e-5,
        learning_rate="adaptive",
        max_iter=400,
        random_state=1,
        early_stopping=False,
    )


def _make_gbt():
    """Paper: 100 trees, depth=3, lr=0.1, 15 random features per split."""
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        max_features=15,
        random_state=1,
    )


def _make_raf():
    """Paper: 1000 trees, depth=20, mRAF=sqrt(31)~6 features per split."""
    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=1,
    )


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class MLPipeline:
    """
    Mirrors CointegrationPipeline's external interface:
        fit(train_prices)            -> trains models, stores warmup prefix
        backtest(test_prices, plot)  -> returns (equity_series, None)

    Parameters
    ----------
    k                  : long + short leg size (default 10; use 5 for Nifty 100)
    total_capital      : starting / carried-forward capital (set by runner each window)
    models             : which base learners to ensemble ("dnn", "gbt", "raf")
    min_prob_threshold : minimum mean P(outperform) of top-k required to trade on a day.
                         Implements the paper's middle-censoring. 0.55 = soft threshold.
                         Set to 0.5 to disable.
    """

    def __init__(
        self,
        k=10,
        total_capital=100.0,
        models=("dnn", "gbt", "raf"),
        min_prob_threshold=0.55,
    ):
        self.k                  = k
        self.total_capital      = total_capital
        self.model_names        = models
        self.min_prob_threshold = min_prob_threshold

        self._dnn            = None
        self._gbt            = None
        self._raf            = None
        self._scaler         = StandardScaler()
        self._fitted         = False
        self._warmup_prices  = None   # last _MAX_LAG rows of training data

    # ------------------------------------------------------------------
    def fit(self, train_prices):
        """
        1. Store last _MAX_LAG days of train_prices as warmup prefix.
        2. Build (X, y) from the full training window.
        3. Fit all enabled models on the scaled feature matrix.
        """
        # Always store warmup even if fit fails (guards against empty windows)
        self._warmup_prices = train_prices.iloc[-_MAX_LAG:].copy()

        print("  Building feature matrix ...", end=" ", flush=True)
        X, y = _compute_features(train_prices)
        if X.empty:
            print("no data -- skipping fit.")
            self._fitted = False
            return self

        print(f"{len(X):,} obs, class balance {y.mean():.2%}")

        X_arr = self._scaler.fit_transform(X.values)

        if "dnn" in self.model_names:
            print("  Training DNN ...", end=" ", flush=True)
            self._dnn = _make_dnn()
            self._dnn.fit(X_arr, y.values)
            print("done.")

        if "gbt" in self.model_names:
            print("  Training GBT ...", end=" ", flush=True)
            self._gbt = _make_gbt()
            self._gbt.fit(X_arr, y.values)
            print("done.")

        if "raf" in self.model_names:
            print("  Training RAF ...", end=" ", flush=True)
            self._raf = _make_raf()
            self._raf.fit(X_arr, y.values)
            print("done.")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _ensemble_proba(self, X_raw):
        """
        Equal-weight ensemble of P(outperform) across all fitted models.
        Paper eq. (5): P_ENS = (P_DNN + P_GBT + P_RAF) / 3.
        """
        X_scaled = self._scaler.transform(X_raw)
        probs = []

        if "dnn" in self.model_names and self._dnn is not None:
            probs.append(self._dnn.predict_proba(X_scaled)[:, 1])
        if "gbt" in self.model_names and self._gbt is not None:
            probs.append(self._gbt.predict_proba(X_scaled)[:, 1])
        if "raf" in self.model_names and self._raf is not None:
            probs.append(self._raf.predict_proba(X_scaled)[:, 1])

        if not probs:
            raise RuntimeError("No fitted models available for prediction.")

        return np.mean(probs, axis=0)

    # ------------------------------------------------------------------
    def backtest(self, test_prices, plot=False):
        """
        Walk day-by-day through test_prices:
          1. Prepend warmup prefix so feature lookback is satisfied from day 1.
          2. For each date, build prediction matrix, rank stocks by P(outperform).
          3. Confidence gate: hold cash if mean(top-k probs) < min_prob_threshold.
          4. Otherwise: long top-k, short bottom-k, dollar-neutral equal weight.
          5. Realise P&L from next-day returns; compound capital.

        Signature matches CointegrationPipeline.backtest(test_prices, plot=False)
        so the shared backtests.walk_forward_pipeline runner needs no changes.

        Returns
        -------
        (equity_series, None)
        """
        if not self._fitted:
            print("  Pipeline not fitted -- holding cash.")
            return pd.Series(self.total_capital, index=test_prices.index), None

        k       = self.k
        capital = self.total_capital   # runner sets pipeline.total_capital before calling

        # -------------------------------------------------------------- #
        # Build price series with warmup prefix prepended.
        # Warmup rows are only used for feature lookback; equity tracking
        # starts at test_prices.index[0].
        # -------------------------------------------------------------- #
        if self._warmup_prices is not None:
            shared_cols        = test_prices.columns.intersection(self._warmup_prices.columns)
            prices_with_warmup = pd.concat([
                self._warmup_prices[shared_cols],
                test_prices[shared_cols],
            ])
            prices_with_warmup = prices_with_warmup[
                ~prices_with_warmup.index.duplicated(keep="last")
            ]
        else:
            prices_with_warmup = test_prices

        test_dates       = test_prices.index
        next_day_returns = test_prices.pct_change(1)

        equity       = [capital]
        equity_dates = [test_dates[0]]
        days_traded  = 0
        days_cash    = 0

        for i, date in enumerate(test_dates[:-1]):
            next_date = test_dates[i + 1]

            X_pred, tickers = _build_predict_matrix(prices_with_warmup, date)

            if X_pred.empty or len(tickers) < 2 * k:
                equity.append(equity[-1])
                equity_dates.append(next_date)
                days_cash += 1
                continue

            probs   = self._ensemble_proba(X_pred.values)
            ranking = pd.Series(probs, index=tickers).sort_values(ascending=False)

            long_stocks  = ranking.index[:k].tolist()
            short_stocks = ranking.index[-k:].tolist()

            # Confidence gate: censor uncertain days (paper's middle-censoring intent)
            if ranking.iloc[:k].mean() < self.min_prob_threshold:
                equity.append(equity[-1])
                equity_dates.append(next_date)
                days_cash += 1
                continue

            # Dollar-neutral: equal position size across all 2k legs
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
                f"threshold={self.min_prob_threshold}, k={k})"
            )

        return pd.Series(equity, index=equity_dates), None
