"""
ML-based statistical arbitrage pipeline for Nifty 100.

Faithful implementation of Krauss, Do & Huck (2016):
  - Features : R(m) for m in {1..20, 40, 60, ..., 240}  →  31 lagged-return features per stock
  - Label    : 1 if stock beats cross-sectional median next-day return, else 0
  - Models   : DNN (MLP), GBT, RAF, and equal-weight ensemble of predicted probabilities
  - Trading  : rank all stocks by P(outperform), long top-k, short bottom-k (dollar-neutral)
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature configuration (paper Section 4.2)
# ---------------------------------------------------------------------------
_LAGS = list(range(1, 21)) + list(range(40, 241, 20))   # 31 features


def _compute_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) from a price DataFrame.

    For every stock s and every date t (where all lags are available):
        X[t, s] = [R(1), R(2), ..., R(20), R(40), ..., R(240)]  for stock s at time t
        y[t, s] = 1 if R(1)_{t+1} for stock s > cross-sectional median R(1)_{t+1}

    Returns
    -------
    X : pd.DataFrame  shape (n_obs, 31)  — each row is one (date, stock) observation
    y : pd.Series     shape (n_obs,)     — binary label
        Both share a MultiIndex (date, ticker).
    """
    max_lag = max(_LAGS)

    # Simple returns R(m) = P_t / P_{t-m} - 1
    feature_frames = {}
    for m in _LAGS:
        feature_frames[f"R{m}"] = prices.pct_change(m)

    features = pd.concat(feature_frames, axis=1)   # MultiIndex columns: (Rm, ticker)
    features.columns = ["_".join([str(c[0]), str(c[1])]) for c in features.columns]

    # Next-day simple return for each stock
    next_ret = prices.pct_change(1).shift(-1)

    rows_X, rows_y = [], []

    # Only iterate over dates where all features AND the label are available
    valid_dates = features.index[max_lag:]          # need max_lag days of history
    valid_dates = valid_dates[valid_dates.isin(next_ret.dropna(how="all").index)]

    for date in valid_dates:
        feat_row = features.loc[date]               # Series indexed by "Rm_ticker"
        ret_row = next_ret.loc[date].dropna()       # next-day returns, drop NaN stocks

        if ret_row.empty:
            continue

        median_ret = ret_row.median()

        for ticker in ret_row.index:
            # Extract all 31 features for this (date, ticker)
            cols = [f"R{m}_{ticker}" for m in _LAGS]
            vals = feat_row.reindex(cols).values
            if np.any(np.isnan(vals)):
                continue
            rows_X.append((date, ticker, vals))
            rows_y.append(1 if ret_row[ticker] > median_ret else 0)

    if not rows_X:
        return pd.DataFrame(), pd.Series(dtype=int)

    index = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows_X],
                                      names=["date", "ticker"])
    X = pd.DataFrame([r[2] for r in rows_X],
                     index=index,
                     columns=[f"R{m}" for m in _LAGS])
    y = pd.Series(rows_y, index=index, name="label")
    return X, y


def _build_predict_matrix(prices: pd.DataFrame, date) -> tuple[pd.DataFrame, list]:
    """
    Build the feature matrix for ALL stocks at a single prediction date `date`.
    Used during backtesting (one day at a time, no label needed).

    Returns
    -------
    X_pred : pd.DataFrame  shape (n_stocks, 31)
    tickers : list of tickers (same order as rows)
    """
    max_lag = max(_LAGS)
    # Need prices up to `date` (inclusive) going back max_lag days
    loc = prices.index.get_loc(date)
    if loc < max_lag:
        return pd.DataFrame(), []

    window = prices.iloc[loc - max_lag: loc + 1]   # shape (max_lag+1, n_stocks)

    rows, tickers = [], []
    for ticker in prices.columns:
        s = window[ticker].dropna()
        if len(s) < max_lag + 1:
            continue
        vals = np.array([s.iloc[-1] / s.iloc[-1 - m] - 1 for m in _LAGS])
        if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
            continue
        rows.append(vals)
        tickers.append(ticker)

    if not rows:
        return pd.DataFrame(), []

    X_pred = pd.DataFrame(rows, index=tickers, columns=[f"R{m}" for m in _LAGS])
    return X_pred, tickers


# ---------------------------------------------------------------------------
# Model definitions (paper Section 4.3)
# ---------------------------------------------------------------------------

def _make_dnn():
    """
    31-31-10-5-2 architecture with dropout approximated via sklearn's alpha (L2) and
    early_stopping.  sklearn MLP does not expose per-layer dropout, so we use:
      - hidden_layer_sizes = (31, 10, 5)  — matches paper topology minus input/output
      - activation = 'relu'  (closest practical equivalent to maxout in sklearn)
      - alpha = 1e-5  (L1/L2 regularisation, paper uses lambda=0.00001)
      - solver = 'adam'  (closest to ADADELTA adaptive learning)
      - max_iter = 400 epochs  (paper trains 400 epochs)
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
    """
    Paper: MGBT=100 trees, JGBT=3 depth, lambda=0.1 learning rate, mGBT=15 features.
    """
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        max_features=15,
        random_state=1,
    )


def _make_raf():
    """
    Paper: BRAF=1000 trees, JRAF=20 depth, mRAF=sqrt(p)=sqrt(31)~6 features.
    """
    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        max_features="sqrt",
        random_state=1,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class MLPipeline:
    """
    Stateful pipeline that mirrors the interface of CointegrationPipeline:
        .fit(train_prices)   →  trains DNN, GBT, RAF on the training window
        .predict_proba(X)    →  returns ensemble probability for each stock
        .backtest(test_prices, k, total_capital)  →  equity_series
    """

    def __init__(self,
                 k: int = 10,
                 total_capital: float = 100.0,
                 models: tuple = ("dnn", "gbt", "raf")):
        self.k = k
        self.total_capital = total_capital
        self.model_names = models

        self._dnn = None
        self._gbt = None
        self._raf = None
        self._scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, train_prices: pd.DataFrame):
        """
        Build features from train_prices, fit all enabled models.
        """
        print("  Building feature matrix …", end=" ", flush=True)
        X, y = _compute_features(train_prices)
        if X.empty:
            print("no data — skipping fit.")
            self._fitted = False
            return self

        print(f"{len(X):,} observations, class balance {y.mean():.2%}")

        X_scaled = self._scaler.fit_transform(X.values)

        if "dnn" in self.model_names:
            print("  Training DNN …", end=" ", flush=True)
            self._dnn = _make_dnn()
            self._dnn.fit(X_scaled, y.values)
            print("done.")

        if "gbt" in self.model_names:
            print("  Training GBT …", end=" ", flush=True)
            self._gbt = _make_gbt()
            self._gbt.fit(X_scaled, y.values)
            print("done.")

        if "raf" in self.model_names:
            print("  Training RAF …", end=" ", flush=True)
            self._raf = _make_raf()
            self._raf.fit(X_scaled, y.values)
            print("done.")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _predict_proba_from_matrix(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Returns ensemble probability P(outperform) for each row in X_raw.
        Ensemble = equal-weight average of all enabled models (paper eq. 5).
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
            raise RuntimeError("No models have been trained.")

        return np.mean(probs, axis=0)

    # ------------------------------------------------------------------
    def backtest(self,
                 test_prices: pd.DataFrame,
                 plot: bool = False) -> tuple:
        """
        Walk day-by-day through test_prices:
          1. Build feature matrix for all stocks at date t
          2. Predict P(outperform) with the ensemble
          3. Long top-k, short bottom-k  (dollar-neutral)
          4. Realise P&L from t→t+1 actual returns

        Signature matches CointegrationPipeline.backtest(test_prices, plot=False)
        so the shared backtests.walk_forward_pipeline runner works unchanged.

        Returns
        -------
        (equity_series, None) — second element is None (no pair-level detail for ML)
        """
        if not self._fitted:
            print("  Pipeline not fitted — returning flat equity.")
            return pd.Series(self.total_capital, index=test_prices.index), None

        k       = self.k
        capital = self.total_capital          # set by walk_forward runner before call

        dates = test_prices.index
        equity = [capital]
        equity_dates = [dates[0]]

        next_day_returns = test_prices.pct_change(1)   # precompute

        for i, date in enumerate(dates[:-1]):           # can't trade on last day
            next_date = dates[i + 1]

            X_pred, tickers = _build_predict_matrix(test_prices, date)
            if X_pred.empty or len(tickers) < 2 * k:
                # Not enough stocks — hold cash
                equity.append(equity[-1])
                equity_dates.append(next_date)
                continue

            probs = self._predict_proba_from_matrix(X_pred.values)
            ranking = pd.Series(probs, index=tickers).sort_values(ascending=False)

            long_stocks  = ranking.index[:k].tolist()
            short_stocks = ranking.index[-k:].tolist()

            # Dollar-neutral: allocate capital / (2k) per position
            pos_size = equity[-1] / (2 * k)

            # Realise returns: long gains if stock goes up, short gains if goes down
            day_ret = next_day_returns.loc[next_date]

            long_pnl  = sum(pos_size * day_ret.get(t, 0.0) for t in long_stocks)
            short_pnl = sum(-pos_size * day_ret.get(t, 0.0) for t in short_stocks)

            new_capital = equity[-1] + long_pnl + short_pnl
            equity.append(max(new_capital, 0.0))        # floor at zero
            equity_dates.append(next_date)

        return pd.Series(equity, index=equity_dates), None
