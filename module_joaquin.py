# module joaquin

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Sequence


def rolling_pca(df: pd.DataFrame,
                window_size: int,
                recalibration_period: int,
                columns: Optional[Sequence] = None):
    if columns:
        df2 = df[columns].copy()
    else:
        df2 = df.copy()

    n_mats = len(df2.columns)

    df2.index = df2.index.strftime('%Y-%m-%d')
    n_obs = len(df2)
    pca_collection = {}
    all_scores = []

    for i in range(window_size, n_obs, recalibration_period):
        key = df2.index[i]
        X = df2.iloc[i - window_size:i, :].to_numpy()
        mean = X.mean(axis=0)
        pca = PCA(n_components=3).fit(X - mean)
        loadings = pca.components_  # shape (3, n_mats)
        if i + recalibration_period <= n_obs:
            Y = df2.iloc[i - 1:i + recalibration_period, :].to_numpy() # shape (recalibration_period, n_mats)
            scores = Y @ loadings.T # shape (recalibration_period, 3)
        else:
            Y = df2.iloc[i - 1:n_obs, :].to_numpy()  # shape (remaining, n_mats)
            scores = Y @ loadings.T  # shape (recalibration_period, 3)
        all_scores.append(scores)


        pca_collection[key] = {
            "explained_var_ratio": pca.explained_variance_ratio_,
            "loadings": pca.components_,  # matrix 3xn_mats
            "mean": mean,
            "scores": scores,
        }

    scores = np.vstack(all_scores)

    return scores, pca_collection


def interest_rate_pca(df, n_obs, rolling_size):
    df.index = df.index.strftime('%Y-%m-%d')
    daily_pca = {}
    for i in range(rolling_size, n_obs):
        key = df.index[i]
        # fit a PCA with 252-days rolling window, using up to date t-1, saved as pca at time t.
        mean = df.iloc[i - rolling_size:i, :].mean()
        pca = PCA(n_components=3).fit(df.iloc[i - rolling_size:i, :] - mean)
        # scores = pca.transform(spot_diff.loc[key].to_numpy()[None,:])[0] #equally valid
        scores = pca.components_ @ df.loc[key].to_numpy()[:, None]
        daily_pca[key] = {
            "explained_var_ratio": pca.explained_variance_ratio_,
            "loadings": pca.components_,  # matrix 3xn_mats
            "mean": mean,
            "scores": scores,
            "level": scores[0],
            "slope": scores[1],
            "curvature": scores[2],
        }
    df.index = pd.to_datetime(df.index)  # Pandas df are called by reference
    return daily_pca
