# module joaquin

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Sequence


def rolling_pca(df: pd.DataFrame,
                corr_window_size: int,
                vol_window_size: int,
                recalibration_period: int,
                columns: Optional[Sequence[str]] = None,
                is_diff: bool = True,
                ):
    if columns is not None:
        df2 = df[columns].copy()
    else:
        df2 = df.copy()

    n_mats = len(df2.columns)
    n_obs = len(df2)
    pca_collection = {}
    all_loadings = []
    all_scores = []
    all_index = []

    if n_obs < corr_window_size:
        raise ValueError("window_size bigger than number of observations.")
    if vol_window_size > corr_window_size:
        raise ValueError("Volatility window should be smaller than correlation window.")
    if recalibration_period > vol_window_size:
        raise ValueError("Recalibration period should be smaller than volatility window.")

    for i in range(corr_window_size, n_obs, recalibration_period):
        key = df2.index[i].strftime("%Y-%m-%d")
        # using corr window
        X = df2.iloc[i - corr_window_size:i, :].to_numpy()
        corr = np.corrcoef(X.T)  # shape (n_mats, n_mats)
        # using vol window
        vol = np.std(X[corr_window_size - vol_window_size: corr_window_size, :], axis=0, ddof=1)
        mean = np.mean(X[corr_window_size - vol_window_size: corr_window_size, :], axis=0)
        cov = vol.T * corr * vol

        if np.any(~np.isfinite(corr)):
            raise ValueError("Correlation matrix not finite")

        if np.any(~np.isfinite(vol)):
            raise ValueError("Volatility vector not finite")

        U, S, U_transpose = np.linalg.svd(cov)
        explained_variance = S[:3] / S.sum()
        loadings = U[:, :3].T  # first three components; shape (3, n_mats)

        start = i - 1
        end = min(i + recalibration_period - 1, n_obs)
        Y = df2.iloc[start:end, :].to_numpy()

        # make the sign correction based on the training sample X
        sign_correction(loadings=loadings,
                        scores=(X - mean) @ loadings.T,
                        X=X,
                        df=df2)

        # second layer of sign corrections: maximize the similarity with the previous PCA loadings.
        if all_loadings:
            for j in range(3):
                if np.dot(loadings[j, :], all_loadings[-1][j, :]) < 0:
                    loadings[j, :] *= -1

        scores = (Y - mean) @ loadings.T
        # scores = pca.transform(Y)  # shape (recalibration_period, 3)
        score_index = df2.index[start:end]

        all_loadings.append(loadings)
        all_scores.append(scores)
        all_index.extend(score_index)

        pca_collection[key] = {
            "explained_var_ratio": explained_variance,
            "loadings": loadings,
            "mean": mean,
            "vol": vol,
            "score_index": score_index,
            "scores": scores,
        }

    scores = np.vstack(all_scores)
    scores_df = pd.DataFrame(scores,
                             index=pd.Index(all_index, name="date"),
                             columns=["PC1", "PC2", "PC3"])

    return scores_df, pca_collection


def sign_correction(loadings: np.ndarray,
                    scores: np.ndarray,
                    X: np.ndarray,
                    df: pd.DataFrame):
    """
    Align PCA signs so that PCs have a stable economic meaning.
    Works whether PCA was run on levels or on differences of rates,
    as long as X and df.columns correspond to the same maturities.

    Convention:
      PC1 (level):  positive => higher overall rates (or positive parallel shock)
      PC2 (slope):  positive => steepening (10Y up relative to 2Y)
      PC3 (curv):   positive => more curvature (5Y up vs 2Y/10Y)
    """

    # indices of rates.
    def _find_col(df, candidates):
        for name in candidates:
            if name in df.columns:
                return df.columns.get_loc(name)
        raise KeyError(f"None of {candidates} found in df.columns")

    idx_2y = _find_col(df, ["USD1y1y", "2Y"])
    idx_5y = _find_col(df, ["USD4y1y", "5Y"])
    idx_10y = _find_col(df, ["USD7y3y", "10Y"])

    # PC1: Level
    avg_delta = X.mean(axis=1)  # cross-sectional avg change each day
    if np.corrcoef(scores[:, 0], avg_delta)[0, 1] < 0:
        loadings[0, :] *= -1

    # PC2: Slope
    slope_scores = scores[:, 1]
    short = X[:, idx_2y]
    long = X[:, idx_10y]
    d_spread = long - short  # delta(10Y−2Y)
    if np.corrcoef(slope_scores, d_spread)[0, 1] < 0:
        loadings[1, :] *= -1

    # PC3: Curvature
    curv_scores = scores[:, 2]
    belly = X[:, idx_5y]
    wings = 0.5 * (X[:, idx_2y] + X[:, idx_10y])
    d_curv = belly - wings
    if np.corrcoef(curv_scores, d_curv)[0, 1] < 0:
        loadings[2, :] *= -1

    return None


def build_pca_lookup(pca_collection):
    """
    pca_collection: dict[datetime-like -> dict(..., 'loadings': ...)]

    Returns a function lookup(dt) that gives you the loadings corresponding to
    the latest key <= dt, or None if none exists.
    """
    # Build a DateTimeIndex from the keys, sorted
    keys = pd.to_datetime(list(pca_collection.keys()))
    idx = pd.DatetimeIndex(keys).sort_values()

    # Map index positions back to the original dict keys (in case keys are not Timestamps)
    # This keeps the *original* keys, but in sorted-by-time order
    sorted_keys = [k for _, k in sorted(zip(idx, pca_collection.keys()), key=lambda x: x[0])]

    def lookup(dt):
        dt = pd.to_datetime(dt)

        # position of the rightmost value <= dt
        pos = idx.searchsorted(dt, side="right") - 1

        if pos < 0:
            # no key older than or equal to dt
            return None

        key = sorted_keys[pos]
        return pca_collection[key]["loadings"]

    """ USAGE
    # Build once
    pca_lookup = build_pca_lookup(pca_collection)
    loadings = pca_lookup(entry_date)   # this is frozen for that trade
    """

    return lookup


def ema_strat(score: pd.DataFrame,
              pca_collecion: dict,
              df: pd.DataFrame,
              slope_portfolio: np.ndarray,
              short: int = 30,
              long: int = 90,
              threshold: float = 0.0001):
    """
    Mean reversion strategy using two exponentialy weighted moving averages:

      Entry (with threshold):
        - Go Short when price > max(EMA_short, EMA_long) + threshold
        - Go Long  when price < min(EMA_short, EMA_long) - threshold

      Exit (no threshold, exit when back to the level of long MA):
        - Close Long when price >= EMA_long
        - Close Short when price <= EMA_long

    Inputs:
    - score: Time-series of the principal components' score
    - pca_collection: Dictionary that contains the pca models over time. (It might be not necessary)
    - df: Rates dataset.
    - slope_portfolio: shape (tenors,); the DV01-neutral portfolio use to take position on the slope of yield curve.
    """
    # --- parameter checks ---
    short, long = int(short), int(long)
    if short < 1 or long < 1 or short >= long:
        return -np.inf

    score_2 = score.copy()

    # --- indicators ---
    score_2["ema_short"] = score["PC2"].ewm(span=short).mean()
    score_2["ema_long"] = score["PC2"].ewm(span=long).mean()

    hi = score_2[["ema_short", "ema_long"]].max(axis=1)
    lo = score_2[["ema_short", "ema_long"]].min(axis=1)

    prices = score_2["PC2"].values
    hi_values = hi.values
    lo_values = lo.values
    ema_long_vals = score_2["ema_long"].values
    dates = score_2.index.values

    pos = 0
    entry_date = None
    exit_date = None
    pos_trade = 0
    positions = []
    trades = []

    pca_lookup = build_pca_lookup(pca_collecion)

    for px, h, l, ema_l, date in zip(prices, hi_values, lo_values, ema_long_vals, dates):
        # still in warmup: no position
        if np.isnan(ema_l):
            pos = 0
        else:
            # --- entry rules (with threshold) ---
            if pos == 0:
                if px > h + threshold:
                    pos = -1  # enter short
                    entry_date = date
                    pos_trade = -1
                elif px < l - threshold:
                    pos = 1  # enter long
                    entry_date = date
                    pos_trade = 1

            # --- exit rules (threshold = 0) ---
            elif pos == 1:  # currently long
                if px >= ema_l:  # price back above lower band
                    pos = 0
                    exit_date = date
            elif pos == -1:  # currently short
                if px <= ema_l:  # price back below upper band
                    pos = 0
                    exit_date = date

        if entry_date and exit_date:
            loading = pca_lookup(entry_date)  # look for the latest PCA loadigs from pca_collections
            trades.append({"entry_date": entry_date,
                           "exit_date": exit_date,
                           "position": pos_trade,
                           "pc_loading": loading[1, :],
                           "portfolio": slope_portfolio})

            entry_date = None
            exit_date = None
            pos_trade = 0

        positions.append(pos)

    score_2["position"] = positions
    score_2["position"] = score_2["position"].fillna(0)

    # Enforce close the position at the final day
    score_2.loc[score_2.index[-1], "position"] = 0
    if entry_date:
        loading = pca_lookup(entry_date)
        trades.append({"entry_date": entry_date,
                       "exit_date": score_2.index[-1],
                       "position": pos_trade,
                       "pc_loading": loading[1, :],
                       "portfolio": slope_portfolio})

    # compute metrics of the trades
    for trade in trades:
        sub = df[(df.index >= trade["entry_date"]) & (df.index <= trade["exit_date"])].to_numpy()
        trading_days = sub.shape[0]
        trade["days"] = trading_days

        w = trade["portfolio"] * trade["position"]
        mtm = (sub - sub[0]) @ w * 10_000 #mtm in basis points
        trade["mtm"] = np.round(mtm, 2)

        trade["pnl"] = np.round(mtm[-1], 2)# pnl in basis points

        if trading_days > 1:
            vol_trade = np.std(np.diff(mtm), ddof=0)  # same as np.sqrt(w.T @ np.cov(np.diff(sub, axis=0).T) @ w)
            trade["vol"] = round(vol_trade, 3)
        else:
            trade["vol"] = 0.0

    return trades, score_2


def ema_strat_trading_sharpe(scores, pca, df, portfolio, short=30, long=90, threshold=0.0001):
    trades, _ = ema_strat(scores, pca, df, portfolio, short, long, threshold)
    mtm_concat = np.ones(shape=(1,))
    sum_pnl = 0
    sum_days = 0

    for trade in trades:
        mtm_concat = np.concatenate([mtm_concat, trade["mtm"]])
        sum_pnl += trade["pnl"]
        sum_days += trade["days"]

    # unrealistic strategy
    if sum_days <= 10 or len(trades) < 4:
        return -np.inf

    sum_vol = np.std(np.diff(mtm_concat[1:]), ddof=0)

    return sum_pnl / (sum_vol * np.sqrt(sum_days))


##########################################
# Old Functions

def dv01_neutral_portfolio(loading: np.ndarray, rates: pd.Series):
    """
    Projection of the PC2 loading onto DV01-neutral hyperplane.
    Obtained a DV01-neutral portfolio that is as close as possible to PC2 loadings.
    :param loading:
    :param rates:
    :return:
    """
    loading = np.asarray(loading, dtype=float)
    annuity = annuity_vector(rates)
    portfolio = loading * annuity - (loading * annuity).mean()
    return portfolio


def annuity_vector(rates: pd.Series):
    """
    Input rates is a slice of 10 buckets
    ['USD1y1y', 'USD2y1y', 'USD3y1y', 'USD4y1y', 'USD5y2y', 'USD7y3y',
       'USD10y5y', 'USD15y5y', 'USD20y5y', 'USD25y5y']
    Compute a rough approximation of a yield curve to obtain the annuity of these swaps.
    Assumes flat rates between buckets.
    """
    time_interval = np.array([1, 1, 1, 1, 1, 2, 3, 5, 5, 5], dtype=int)
    r = np.asarray(rates)
    expanded_rates = np.repeat(r, time_interval)
    integral_rate = np.cumsum(expanded_rates)
    dis_factor = np.exp(-integral_rate)
    alpha = 1.0
    annuity = np.cumsum(alpha * dis_factor)
    idx = np.cumsum(time_interval) - 1
    return annuity[idx]
