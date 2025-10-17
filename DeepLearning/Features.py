from typing import Sequence, Tuple, Union
import pandas as pd
import numpy as np

def vol_features(log_returns_array: np.ndarray, kernel_array: np.ndarray, sentiment_array: np.ndarray, lookback: int, ma_windows: Sequence[int], horizons: Union[Sequence[int], int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct volatility-related features and multi-horizon targets.

    Builds a feature matrix X from returns, kernel (volatility proxy), and
    categorical sentiment; includes lags up to `lookback`, rolling means/stds
    over `ma_windows`, and rolling sentiment proportions/entropy. Targets Y are
    future kernel values at the specified `horizons`. Rows that would require
    future information are dropped to align X and Y.

    Parameters
    ----------
    log_returns_array : np.ndarray
        1-D array of log returns r_t.
    kernel_array : np.ndarray
        1-D array of kernel values rk_t (target base series).
    sentiment_array : np.ndarray
        1-D array of sentiment class labels (categorical, 3 classes expected).
    lookback : int
        Maximum lag order for features (≥ 0).
    ma_windows : Sequence[int]
        Rolling window sizes for moving-average features.
    horizons : Sequence[int] or int
        Forecast horizons (positive integers). If a single int is provided,
        a single-column Y is produced; otherwise, one column per horizon.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple `(X, Y)` where:
          - `X` has engineered features with shape (N_eff, n_features),
          - `Y` stacks future rk values with shape (N_eff, n_horizons).

    Notes
    -----
    - Sentiment is one-hot encoded; rolling proportions are computed per class.
    - Entropy uses `-sum(p * log p)` without epsilon; zeros will propagate NaNs
      as in the original implementation.
    - The final `N_eff` excludes initial rows lost to lags/rolls and the last
      `max(horizons)` rows needed to form targets.
    """
    df = pd.DataFrame({"r": log_returns_array, "rk": kernel_array, "s": sentiment_array})
    df["abs_r"] = df["r"].abs(); df["r2"] = df["r"] ** 2; df["log_rk"] = np.log(df["rk"])

    sentiment_dummies = pd.get_dummies(df["s"], prefix="s")
    df = pd.concat([df, sentiment_dummies], axis=1)
    sentiment_cols = sentiment_dummies.columns.tolist()

    for i in range(1, lookback + 1):
        df[f"rk_lag{i}"] = df["rk"].shift(i)
        df[f"|r|_lag{i}"] = df["abs_r"].shift(i)
        df[f"r2_lag{i}"] = df["r2"].shift(i)
        for col in sentiment_cols: df[f"{col}_lag{i}"] = df[col].shift(i)

    for w in ma_windows:
        df[f"rk_ma{w}"] = df["rk"].rolling(w).mean()
        df[f"rk_std{w}"] = df["rk"].rolling(w).std(ddof=0)
        df[f"|r|_ma{w}"] = df["abs_r"].rolling(w).mean()
        df[f"r2_ma{w}"] = df["r2"].rolling(w).mean()
        df[f"log_rk_ma{w}"] = df["log_rk"].rolling(w).mean()
        for col in sentiment_cols: df[f"{col}_prop_ma{w}"] = df[col].rolling(w).mean()
        prop_cols = [f"{col}_prop_ma{w}" for col in sentiment_cols]
        df[f"s_entropy_ma{w}"] = -(df[prop_cols].apply(lambda row: (row * np.log(row)).sum(), axis=1))

    df = df.dropna().reset_index(drop=True)

    if isinstance(horizons, (list, tuple)) and len(horizons) > 1:
        Y = np.column_stack([df["rk"].shift(-h) for h in horizons])
        max_h = max(horizons)
    else:
        h0 = horizons[0] if isinstance(horizons, (list, tuple)) else int(horizons)
        Y = df["rk"].shift(-h0).to_frame().to_numpy()
        max_h = h0

    df = df.iloc[:-max_h]
    Y = Y[:-max_h]
    X = df.drop(columns=["rk", "s"]).to_numpy(dtype=float)
    return X, Y
