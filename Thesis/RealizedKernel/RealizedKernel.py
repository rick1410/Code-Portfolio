from typing import Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RealizedKernel:
    """
    Utilities to compute realized-kernel estimates from irregular intraday prices.

    Parameters
    ----------
    data_root_dir : str
        Root directory to which result plots are written.

    Attributes
    ----------
    data_root_dir : str
        Stored output root path.
    """

    data_root_dir: str

    def __init__(self, data_root_dir: str) -> None:
        self.data_root_dir = data_root_dir

    def irregular_prices_log_returns(self, price_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert irregular log-price ticks into daily high-frequency returns.

        Parameters
        ----------
        price_data : pd.DataFrame
            DataFrame indexed by timestamp, containing a column "log prices".

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            Tuple `(returns_df, counts_df)`:
              - `returns_df` indexed by timestamp with columns ["Returns","Date"].
              - `counts_df` indexed by date with column ["NumReturns"].
        """
        df = price_data.sort_index().copy()
        df['Date'] = df.index.date
        daily_returns = []; daily_counts = []
        for date, group in df.groupby('Date'):
            timestamps = group.index
            prices = group["log prices"].values
            n = len(prices)
            X = np.zeros(n, dtype=float)
            X[0] = 0.5 * (prices[0] + prices[1])
            X[1:-1] = prices[1:-1]
            X[-1] = 0.5 * (prices[-2] + prices[-1])
            returns = np.diff(X)
            daily_returns.append(pd.DataFrame({"Timestamp": timestamps[1:], "Returns": returns, "Date": date}))
            daily_counts.append({"Date": date, "NumReturns": len(returns)})
        returns_df = pd.concat(daily_returns).set_index("Timestamp")
        counts_df = pd.DataFrame(daily_counts)
        counts_df['Date'] = pd.to_datetime(counts_df['Date'])
        counts_df.set_index("Date", inplace=True)
        return returns_df, counts_df

    def compute_shifted_rvs_multi_day(self, log_prices: pd.Series, window_size: int = 1200, shifts: int = 1200) -> pd.DataFrame:
        """
        Compute per-day sparse realized variance using non-overlapping windows.

        Parameters
        ----------
        log_prices : pd.Series
            Intraday log prices indexed by timestamp.
        window_size : int, default 1200
            Window length (number of ticks) between endpoints.
        shifts : int, default 1200
            Unused placeholder kept for API compatibility.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by Date with column "RV_Sparse".
        """
        s = log_prices.copy()
        s.index = pd.to_datetime(s.index)
        grouped = s.groupby(s.index.date)
        daily_rv_sparse = {}
        for day, daily_prices in grouped:
            n = len(daily_prices)
            shifted_returns = []
            for start in range(0, n - window_size, window_size):
                end = start + window_size
                log_return = daily_prices.iloc[end] - daily_prices.iloc[start]
                shifted_returns.append(log_return)
            rv_sparse = np.sum(np.square(shifted_returns))
            daily_rv_sparse[day] = rv_sparse
        rv_sparse_df = pd.DataFrame(list(daily_rv_sparse.items()), columns=['Date', 'RV_Sparse'])
        rv_sparse_df['Date'] = pd.to_datetime(rv_sparse_df['Date'])
        rv_sparse_df.set_index("Date", inplace=True)
        return rv_sparse_df

    def compute_omega_squared(self, log_prices: pd.Series, q: int) -> pd.DataFrame:
        """
        Estimate microstructure noise variance ω² per day via dense RV.

        Parameters
        ----------
        log_prices : pd.Series
            Intraday log prices indexed by timestamp.
        q : int
            Sampling step (ticks) between endpoints.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by Date with column "Omega_squared".
        """
        s = log_prices.copy()
        s.index = pd.to_datetime(s.index)
        grouped = s.groupby(s.index.date)
        omega_squared_daily = {}
        for day, daily_prices in grouped:
            n = len(daily_prices)
            returns = []
            for i in range(0, n - q, q):
                start = i; end = i + q
                log_return = daily_prices.iloc[end] - daily_prices.iloc[start]
                returns.append(log_return)
            rv_dense = np.sum(np.square(returns))
            num_nonzero = len(returns)
            omega_squared = rv_dense / (2 * num_nonzero) if num_nonzero > 0 else np.nan
            omega_squared_daily[day] = omega_squared
        omega_squared_df = pd.DataFrame(list(omega_squared_daily.items()), columns=['Date', 'Omega_squared'])
        omega_squared_df['Date'] = pd.to_datetime(omega_squared_df['Date'])
        omega_squared_df.set_index("Date", inplace=True)
        return omega_squared_df

    @staticmethod
    def parzen_kernel(x: float) -> float:
        """
        Parzen kernel value at x in [-1, 1].

        Parameters
        ----------
        x : float
            Scaled lag.

        Returns
        -------
        float
            Kernel weight.
        """
        ax = abs(x)
        if 0 <= ax <= 0.5:
            return float(1 - 6 * ax**2 + 6 * ax**3)
        elif 0.5 < ax <= 1:
            return float(2 * (1 - ax) ** 3)
        else:
            return 0.0

    def estimate_bandwidth_per_day(self, omega_squared_df: pd.DataFrame, rv_sparse_df: pd.DataFrame, daily_counts_df: pd.DataFrame, c_star: float = 3.5134) -> pd.DataFrame:
        """
        Compute optimal bandwidth H* per day using xi² and sample size n.

        Parameters
        ----------
        omega_squared_df : pd.DataFrame
            Per-day ω² estimates with column "Omega_squared".
        rv_sparse_df : pd.DataFrame
            Per-day sparse RV with column "RV_Sparse".
        daily_counts_df : pd.DataFrame
            Per-day counts with column "NumReturns".
        c_star : float, default 3.5134
            Constant from asymptotic theory.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by Date with column "H_star" (int).
        """
        bandwidth_per_day = {}
        for date, row in omega_squared_df.iterrows():
            omega_squared = row['Omega_squared']
            sparse_rv = rv_sparse_df.loc[date, 'RV_Sparse']
            n = daily_counts_df.loc[date, 'NumReturns']
            xi_squared = omega_squared / sparse_rv if sparse_rv != 0 else np.nan
            H_star = c_star * (xi_squared ** (4 / 5)) * (n ** (3 / 5)) if np.isfinite(xi_squared) else np.nan
            H_star = int(np.ceil(H_star)) if np.isfinite(H_star) else 0
            bandwidth_per_day[date] = H_star
        bandwidth_df = pd.DataFrame(list(bandwidth_per_day.items()), columns=['Date', 'H_star'])
        bandwidth_df['Date'] = pd.to_datetime(bandwidth_df['Date'])
        bandwidth_df.set_index('Date', inplace=True)
        return bandwidth_df

    def realized_kernel(self, returns: np.ndarray, bandwidth: int) -> float:
        """
        Compute realized-kernel estimate using Parzen weights and sample autocovariances.

        Parameters
        ----------
        returns : np.ndarray
            High-frequency returns within a day (1-D array).
        bandwidth : int
            Kernel bandwidth H (nonnegative integer).

        Returns
        -------
        float
            Realized-kernel estimate.
        """
        n = len(returns)
        def gamma_h(h: int) -> float:
            if h > 0:
                return float(np.sum(returns[h:] * returns[:n - h]))
            else:
                return float(np.sum(returns ** 2))
        H = int(bandwidth)
        kernel_values = [self.parzen_kernel(h / (H + 1)) for h in range(-H, H + 1)]
        realized_covariances = [gamma_h(abs(h)) for h in range(-H, H + 1)]
        kernel_estimate = float(np.sum(np.array(kernel_values) * np.array(realized_covariances)))
        return kernel_estimate

    def calculate_daily_realized_kernels(self, daily_returns_df: pd.DataFrame, bandwidth_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute realized-kernel per day given intraday returns and per-day bandwidth.

        Parameters
        ----------
        daily_returns_df : pd.DataFrame
            DataFrame with columns ["Returns","Date"] indexed by intraday timestamps.
        bandwidth_df : pd.DataFrame
            DataFrame indexed by Date with column "H_star".

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by Date with column "RealizedKernel".
        """
        realized_kernels = []
        grouped_returns = daily_returns_df.groupby(daily_returns_df['Date'])
        for date, group in grouped_returns:
            date = pd.Timestamp(date)
            returns = group['Returns'].values
            bandwidth = int(bandwidth_df.loc[date, 'H_star'])
            kernel_estimate = self.realized_kernel(returns, bandwidth)
            realized_kernels.append({'Date': date, 'RealizedKernel': kernel_estimate})
        realized_kernels_df = pd.DataFrame(realized_kernels)
        realized_kernels_df['Date'] = pd.to_datetime(realized_kernels_df['Date'])
        realized_kernels_df.set_index('Date', inplace=True)
        return realized_kernels_df

    def merge_and_plot_realized_data(self, realized_kernels_df: pd.DataFrame, realised_vol: pd.Series, log_returns: pd.Series, stock: str) -> pd.DataFrame:
        """
        Merge realized-kernel, realized volatility, and returns; save a PNG plot.

        Parameters
        ----------
        realized_kernels_df : pd.DataFrame
            DataFrame indexed by Date with column "RealizedKernel".
        realised_vol : pd.Series
            Per-day realized volatility.
        log_returns : pd.Series
            Per-day log returns.
        stock : str
            Stock ticker (used in title and output path).

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with columns ["Kernel","Volatility","log returns"].
        """
        vol_df = realised_vol.rename("Volatility").to_frame()
        returns_df = log_returns.rename("log returns").to_frame()
        if isinstance(returns_df.index, pd.DatetimeIndex) and returns_df.index.tz is not None:
            returns_df.index = returns_df.index.tz_localize(None)
        merged_df = realized_kernels_df.join(vol_df, how="inner").rename(columns={"RealizedKernel": "Kernel"})
        merged_df = merged_df.join(returns_df, how="inner")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(merged_df.index, merged_df['Kernel'], label='Realized Kernel', linewidth=2)
        ax.plot(merged_df.index, merged_df['Volatility'], label='Realized Volatility', linewidth=2)
        years = pd.date_range(merged_df.index.min(), merged_df.index.max(), freq='YS')
        ax.set_xticks(years); ax.set_xticklabels([y.year for y in years])
        ax.set_xlabel('Year', fontsize=12); ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{stock}: Realized Kernel & Volatility Over Time', fontsize=14)
        ax.legend(loc='upper left'); ax.grid(alpha=0.3); fig.tight_layout()
        results_dir = os.path.join(self.data_root_dir, stock, f"{stock} Results")
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{stock}_kernel_vs_volatility.png")
        fig.savefig(out_path, dpi=350); plt.close(fig)
        return merged_df
