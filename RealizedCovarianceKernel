# RealizedCovarianceKernel.py
from __future__ import annotations

from typing import Dict, Iterable, Tuple, Optional, Callable
import numpy as np
import pandas as pd
import numba
from numba import njit, prange

# =========================
# Numba helpers (module-level)
# =========================

@njit
def _refresh_time_numba(indeces, values):
    """
    Merge two or more irregular timestamp arrays into a common grid using
    'refresh time' logic and carry forward only when all series have traded
    since the last grid point.
    """
    # build union of all timestamps
    merged_index = indeces[0]
    for idx in indeces[1:]:
        merged_index = np.append(merged_index, idx)
    # unique + sort
    merged_index = np.unique(merged_index)

    # initialize output
    merged_values = np.empty((merged_index.shape[0], len(values)))
    merged_values[:, :] = np.nan

    last_values = np.empty(merged_values.shape[1])
    last_values[:] = np.nan

    for i in range(merged_index.shape[0]):
        t = merged_index[i]
        for j in range(merged_values.shape[1]):
            index_j = indeces[j]
            vals_j  = values[j]
            # position where t would be inserted to keep order
            loc = np.searchsorted(index_j, t)
            # guard bounds BEFORE indexing
            if loc < index_j.shape[0] and index_j[loc] == t:
                # safe to read vals_j[loc]
                last_values[j] = vals_j[loc]

        # commit a grid point only when every series traded at least once
        if not np.isnan(last_values).any():
            merged_values[i, :] = last_values
            # reset the "since last refresh" memory
            for k in range(last_values.shape[0]):
                last_values[k] = np.nan

    return merged_values, merged_index


@njit
def _upper_triangular_indeces(p: int) -> np.ndarray:
    s = 0
    idx = np.zeros((int((p * (p + 1) / 2)), 2), dtype=np.int16)
    for i in range(p):
        for j in range(i, p):
            idx[s, 0] = i
            idx[s, 1] = j
            s += 1
    # sanity check against overflow
    if idx[-1, 0] < 0:
        raise ValueError("Got negative index, 'p' probably too large for int16")
    return idx


@njit
def _parzen_kernel_numba(x: float) -> float:
    ax = np.abs(x)
    if 0.0 <= ax <= 0.5:
        return 1.0 - 6.0 * ax * ax + 6.0 * ax * ax * ax
    elif 0.5 < ax <= 1.0:
        d = 1.0 - ax
        return 2.0 * d * d * d
    else:
        return 0.0


@njit
def _gamma(data: np.ndarray, h: int) -> np.ndarray:
    """
    data: (p, n)
    returns p x p autocovariance at lag h
    """
    if h == 0:
        gamma_h = data @ data.T
    else:
        ah = abs(h)
        gamma_h = data[:, ah:] @ data[:, :-ah].T
    if h < 0:
        gamma_h = gamma_h.T
    return gamma_h


@njit(cache=False, parallel=False, fastmath=False)
def _krvm_core(data: np.ndarray, H: int, kernel) -> np.ndarray:
    """
    Kernel Realized Variance/Matrix on synchronized data.
    data: (p, n)
    """
    p, n = data.shape
    # positive semi-definite version switch (kept for completeness)
    c = 1
    cov = _gamma(data, 0)
    # only need up to n-1 meaningful lags
    for h in range(1, n):
        w = kernel((h - c) / H)
        if w == 0.0:
            break
        g = _gamma(data, h)
        cov += w * (g + g.T)
    return cov


@njit(cache=False, parallel=True, fastmath=False)
def _krvm_pairwise(indeces: np.ndarray,
                   values: np.ndarray,
                   H: int,
                   kernel) -> np.ndarray:
    """
    Pairwise KRVM on irregular data via refresh-time synchronization.
    indeces, values: (p, n_max) with trailing NaNs in values and
                     corresponding garbage in indeces ignored by lengths.
    """
    p = indeces.shape[0]
    cov = np.ones((p, p))
    idx_ut = _upper_triangular_indeces(p)

    # pre-compute lengths of valid (non-NaN) values per series
    lengths = np.empty(p, dtype=np.int64)
    for i in range(p):
        # count non-nan at the tail (values carry the NaN mask)
        count = 0
        for k in range(values.shape[1]):
            if not np.isnan(values[i, k]):
                count += 1
        lengths[i] = count

    for t in prange(len(idx_ut)):
        i = idx_ut[t, 0]
        j = idx_ut[t, 1]

        n_i = lengths[i]
        n_j = lengths[j]

        if i == j:
            # price increments
            series = values[i, :n_i].reshape(-1, 1)
            data = (series[1:, :] - series[:-1, :]).T  # (1, T)
            cov[i, i] = _krvm_core(data, H, kernel)[0, 0]
        else:
            merged_values, _ = _refresh_time_numba(
                (indeces[i, :n_i], indeces[j, :n_j]),
                (values[i, :n_i], values[j, :n_j])
            )

            # drop rows with any NaN (numba: flatten workaround)
            mv = merged_values.reshape(-1)
            # count non-NaN in pairs
            valid = 0
            for u in range(0, mv.shape[0], 2):
                if not (np.isnan(mv[u]) or np.isnan(mv[u + 1])):
                    valid += 1

            data = np.empty((2, max(0, valid - 1)))
            # build increments
            prev_a = np.nan
            prev_b = np.nan
            pos = 0
            for u in range(0, mv.shape[0], 2):
                a = mv[u]
                b = mv[u + 1]
                if not (np.isnan(a) or np.isnan(b)):
                    if not (np.isnan(prev_a) or np.isnan(prev_b)):
                        da = a - prev_a
                        db = b - prev_b
                        if data.shape[1] > 0 and pos < data.shape[1]:
                            data[0, pos] = da
                            data[1, pos] = db
                            pos += 1
                    prev_a = a
                    prev_b = b

            if data.shape[1] == 0:
                cov[i, j] = cov[j, i] = np.nan
            else:
                c_ij = _krvm_core(data, H, kernel)[0, 1]
                cov[i, j] = c_ij
                cov[j, i] = c_ij

    return cov


# =========================
# High-level Python helpers
# =========================

def parzen_kernel(x: float) -> float:
    # Python version (kept for API symmetry); numba version is used in hot loops.
    ax = abs(x)
    if 0.0 <= ax <= 0.5:
        return 1.0 - 6.0 * ax**2 + 6.0 * ax**3
    elif 0.5 < ax <= 1.0:
        return 2.0 * (1.0 - ax)**3
    return 0.0


def get_bandwidth(n: int, var_ret: float, var_noise: float, c_star: float = 3.5134) -> int:
    xi_sq = var_noise / var_ret
    H = int(c_star * (xi_sq ** (2.0 / 5.0)) * (n ** (3.0 / 5.0)))
    return max(1, H)


def _get_indeces_and_values(tick_series_list: Iterable[pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
    n_max = int(np.max([len(x) for x in tick_series_list])) if len(tick_series_list) else 0
    p = len(tick_series_list)
    indeces = np.empty((p, n_max), dtype=np.uint64)
    values = np.empty((p, n_max), dtype=np.float64)
    indeces[:, :] = 0  # will be ignored beyond valid lengths
    values[:, :] = np.nan
    for i, s in enumerate(tick_series_list):
        s_clean = s.dropna()
        idx = np.array(s_clean.index.values, dtype=np.uint64)
        v = s_clean.to_numpy(dtype=np.float64)
        indeces[i, :idx.shape[0]] = idx
        values[i, :idx.shape[0]] = v
    return indeces, values


def _refresh_time_py(tick_series_list: Iterable[pd.Series]) -> pd.DataFrame:
    indeces = tuple([np.array(x.dropna().index.values, dtype=np.uint64) for x in tick_series_list])
    values  = tuple([x.dropna().to_numpy(dtype=np.float64) for x in tick_series_list])
    rt_data, index = _refresh_time_numba(indeces, values)
    index = pd.to_datetime(index)
    return pd.DataFrame(rt_data, index=index).dropna(how="any")


# =========================
# Main class
# =========================

class RealizedCovarianceKernel:
    """
    Wrapper around KRVM for irregular high-frequency prices with refresh-time
    synchronization + optional daily realized-kernel workflow using a provided
    'rk' module (with your existing helpers).
    """

    def __init__(self,
                 all_irregular_logprice: Dict[str, pd.DataFrame],
                 all_one_sec_logprice: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Parameters
        ----------
        all_irregular_logprice : dict of {ticker: DataFrame/Series}
            Each value must be a 1-column DataFrame or Series of log prices
            indexed by datetime (irregular).
        all_one_sec_logprice : dict, optional
            Same tickers mapped to 1-second regularized log price DataFrames.
            Used for alternate market proxy construction.
        """
        self.all_irregular_logprice = all_irregular_logprice
        self.all_one_sec_logprice = all_one_sec_logprice

        # cache commonly used lists
        self._tick_series_list = self._extract_tick_series(all_irregular_logprice)

    @staticmethod
    def _extract_tick_series(price_dict: Dict[str, pd.DataFrame]) -> list[pd.Series]:
        out = []
        for _, df in price_dict.items():
            s = df.squeeze().dropna().sort_index()
            if not isinstance(s, pd.Series):
                raise ValueError("Each value must be a 1D Series/1-column DataFrame.")
            out.append(s)
        return out

    # ---------- Core KRVM ----------

    def krvm(self,
             H: int,
             kernel: Optional[Callable[[float], float]] = None) -> np.ndarray:
        """
        Compute the kernel realized covariance matrix pairwise from irregular data.
        """
        if kernel is None:
            kernel = _parzen_kernel_numba  # numba version for speed
        indeces, values = _get_indeces_and_values(self._tick_series_list)
        cov = _krvm_pairwise(indeces, values, int(H), kernel)
        return cov

    # ---------- Market proxies & realized vol ----------

    def market_series_from_irregular(self,
                                    resample_rule: str = "5T") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build a market 'index' from refresh-time synchronized prices (mean across names),
        then compute intraday log-returns at `resample_rule` and daily realized volatility.
        """
        refreshed = _refresh_time_py(self._tick_series_list)
        market_prices = refreshed.mean(axis=1).to_frame(name="log prices")

        # interval returns
        interval_prices = market_prices["log prices"].resample(resample_rule).median()
        log_ret = 100.0 * interval_prices.diff().dropna()
        # daily realized variance (sum of squared intraday returns), skip first partial day
        daily_rv = log_ret.groupby(log_ret.index.date).apply(lambda x: np.sum(x.values**2))
        daily_rv = daily_rv.iloc[1:]  # drop first (often partial) day
        return market_prices, daily_rv

    def market_series_from_one_sec(self) -> pd.DataFrame:
        """
        Build a market 'index' from 1-second regular prices (mean across names).
        """
        if self.all_one_sec_logprice is None:
            raise ValueError("all_one_sec_logprice was not provided.")
        s_list = []
        for stock, df in self.all_one_sec_logprice.items():
            s = df.squeeze()
            if isinstance(df, pd.DataFrame) and "log prices" in df.columns:
                s = df["log prices"]
            s_list.append(s.rename(stock))
        df_combined = pd.concat(s_list, axis=1).sort_index()
        return df_combined.mean(axis=1).to_frame(name="log prices")

    # ---------- Realized-kernel workflow via user 'rk' module ----------

    def realized_kernel_workflow(self,
                                 rk_module,
                                 resample_rule: str = "5T",
                                 window_size: int = 1200,
                                 shifts: int = 1200,
                                 q: int = 25,
                                 c_star: float = 3.5134) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Runs your rk.* helpers to compute daily realized kernels and bandwidths
        and merges with a daily realized volatility proxy from refresh-time data.

        Returns
        -------
        merged_df : pd.DataFrame
        bandwidth_df : pd.DataFrame
        realized_kernels_df : pd.DataFrame (scaled by 1e4)
        """
        # market proxies
        market_prices_irreg, market_rv_daily = self.market_series_from_irregular(resample_rule=resample_rule)
        market_prices_1s = self.market_series_from_one_sec() if self.all_one_sec_logprice is not None else market_prices_irreg

        # rk helpers
        daily_returns_df, daily_counts_df = rk_module.irregular_prices_log_returns(market_prices_irreg)
        rv_sparse_df = rk_module.compute_shifted_rvs_multi_day(market_prices_1s, window_size=window_size, shifts=shifts)
        omega_squared_df = rk_module.compute_omega_squared(market_prices_irreg, q=q)
        bandwidth_df = rk_module.estimate_bandwidth_per_day(omega_squared_df, rv_sparse_df, daily_counts_df, c_star=c_star)
        realized_kernels_df = rk_module.calculate_daily_realized_kernels(daily_returns_df, bandwidth_df)
        realized_kernels_df["RealizedKernel"] = realized_kernels_df["RealizedKernel"] * 10000.0

        merged_df = rk_module.merge_and_plot_realized_data(realized_kernels_df, market_rv_daily)

        return merged_df, bandwidth_df, realized_kernels_df

    # ---------- Convenience ----------

    def refresh_time(self) -> pd.DataFrame:
        """
        Return refresh-time synchronized values (each column is a ticker).
        """
        df = _refresh_time_py(self._tick_series_list)
        # add column names (if present in dict)
        names = list(self.all_irregular_logprice.keys())
        if len(names) == df.shape[1]:
            df.columns = names
        return df





