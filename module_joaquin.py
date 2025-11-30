# module joaquin

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Sequence


def rolling_pca(df: pd.DataFrame,
                window_size: int,
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

    if n_obs < window_size:
        raise ValueError("window_size bigger than number of observations")

    for i in range(window_size, n_obs, recalibration_period):
        key = df2.index[i].strftime("%Y-%m-%d")
        X = df2.iloc[i - window_size:i, :].to_numpy()
        pca = PCA(n_components=3).fit(X)
        mean = pca.mean_.copy()
        loadings = pca.components_.copy() # shape (3, n_mats)

        start = i - 1
        end = min(i + recalibration_period - 1, n_obs)
        Y = df2.iloc[start:end, :].to_numpy()

        # make the sign correction based on the training sample X
        sign_correction(loadings = loadings,
                       scores = (X - mean) @ loadings.T,
                       X = X,
                       df = df2)

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
            "explained_var_ratio": pca.explained_variance_ratio_,
            "loadings": loadings,
            "original_loadings": pca.components_,
            "mean": mean,
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
    idx_2y = df.columns.get_loc("2Y")
    idx_5y = df.columns.get_loc("5Y")
    idx_10y = df.columns.get_loc("10Y")

    #PC1: Level
    avg_delta = X.mean(axis=1)  # cross-sectional avg change each day
    if np.corrcoef(scores[:, 0], avg_delta)[0, 1] < 0:
        loadings[0, :] *= -1

    #PC2: Slope
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


