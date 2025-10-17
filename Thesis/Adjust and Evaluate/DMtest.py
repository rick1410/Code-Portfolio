from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests  # kept for parity with original imports

class DM_test:
    """
    Diebold–Mariano (DM) tests for equal predictive accuracy.

    Provides:
      - Univariate pairwise DM tests per horizon using Newey–West variance.
      - Panel DM test (Bartlett kernel HAC) aggregating across tickers.

    Notes
    -----
    - The loss is squared error. For models m1 and m2, the loss difference is
      d_t = (e1_t^2 - e2_t^2), where e_i are forecast errors.
    - Newey–West lag defaults to ⌊T^{1/3}⌋ as in the original implementation.
    """

    @staticmethod
    def newey_west(d: np.ndarray, lag: int | None = None) -> float:
        """
        Newey–West long-run variance of a 1-D series (Bartlett kernel).

        Parameters
        ----------
        d : np.ndarray
            1-D array of loss differences.
        lag : int or None, default None
            Truncation lag; if None, uses ⌊T^{1/3}⌋.

        Returns
        -------
        float
            HAC variance estimate.
        """
        T = len(d)
        d = d - d.mean()
        if lag is None: lag = int(T ** (1 / 3))
        gamma0 = float(np.dot(d, d) / T)
        var = gamma0
        for h in range(1, lag + 1):
            gamma_h = float(np.dot(d[:-h], d[h:]) / T)
            weight = 1 - h / (lag + 1)
            var += 2 * weight * gamma_h
        return float(var)

    @staticmethod
    def calculate_dm_test(realized_kernel: Dict[Any, np.ndarray] | Dict[Any, list], forecasts: Dict[Any, Dict[str, np.ndarray | list]]) -> Dict[Any, pd.DataFrame]:
        """
        Pairwise DM tests across models for each horizon (univariate per ticker).

        Parameters
        ----------
        realized_kernel : dict
            Mapping horizon -> realized kernel array-like (length T).
        forecasts : dict
            Mapping horizon -> dict(model_name -> forecast array-like of length T).

        Returns
        -------
        dict[Any, pd.DataFrame]
            Per-horizon DataFrame with columns ["Model 1","Model 2","DM Statistic","P-Value"].
        """
        dm_results: Dict[Any, pd.DataFrame] = {}
        lag_fn = lambda T: int(T ** (1 / 3))
        for h in forecasts:
            model_names = list(forecasts[h].keys())
            rows = []
            T = len(realized_kernel[h])
            lag = lag_fn(T)
            for model1, model2 in combinations(model_names, 2):
                e1 = np.asarray(forecasts[h][model1]) - np.asarray(realized_kernel[h])
                e2 = np.asarray(forecasts[h][model2]) - np.asarray(realized_kernel[h])
                d = e1**2 - e2**2
                var = DM_test.newey_west(d, lag)
                dm_stat = float(d.mean() / np.sqrt(var / T))
                p_val = float(2 * (1 - norm.cdf(abs(dm_stat))))
                rows.append({"Model 1": model1, "Model 2": model2, "DM Statistic": dm_stat, "P-Value": p_val})
            dm_results[h] = pd.DataFrame(rows)
        return dm_results

    @staticmethod
    def panel_dm_bartlett(loss_diff: np.ndarray) -> Tuple[float, float]:
        """
        Panel DM statistic using Bartlett HAC across time, averaged across series.

        Parameters
        ----------
        loss_diff : np.ndarray
            Matrix of loss differences with shape (T, n), columns are model-pair
            differences across tickers.

        Returns
        -------
        (float, float)
            (DM statistic, two-sided p-value).
        """
        T, n = loss_diff.shape
        R_raw = np.sqrt(n) * loss_diff.mean(axis=1)
        num = float(R_raw.sum())
        R_c = R_raw - R_raw.mean()
        var_hac = DM_test.newey_west(R_c)
        dm_stat = float(num / np.sqrt(T * var_hac))
        p_val = float(2 * (1 - norm.cdf(abs(dm_stat))))
        return dm_stat, p_val

    @staticmethod
    def panel_dm(results_by_ticker: Dict[str, Any], alpha: float = 0.05) -> Dict[Any, pd.DataFrame]:
        """
        Panel DM tests across tickers for all model pairs at each horizon.

        Parameters
        ----------
        results_by_ticker : dict
            For each ticker, a dict with key 'combined' containing:
              - 'realized_kernel': dict[horizon] -> array-like (T,)
              - 'forecasts': dict[horizon] -> dict[model] -> array-like (T,)
        alpha : float, default 0.05
        Returns
        -------
        dict[Any, pd.DataFrame]
            Per-horizon DataFrame with columns ["Model 1","Model 2","DM Statistic","P-Value"].
        """
        sample = next(iter(results_by_ticker.values()))['combined']
        horizons = sorted(sample['realized_kernel'].keys())
        models = sorted({model_name for res in results_by_ticker.values() for h, fdict in res['combined']['forecasts'].items() for model_name in fdict.keys()})
        dm_by_horizon: Dict[Any, pd.DataFrame] = {}
        for h in horizons:
            rows = []
            for m1, m2 in combinations(models, 2):
                diffs = []
                for _, res in results_by_ticker.items():
                    rk = np.asarray(res['combined']['realized_kernel'][h])
                    f1 = np.asarray(res['combined']['forecasts'][h].get(m1))
                    f2 = np.asarray(res['combined']['forecasts'][h].get(m2))
                    diffs.append((f1 - rk) ** 2 - (f2 - rk) ** 2)
                loss_diff = np.column_stack(diffs)
                dm_stat, p_val = DM_test.panel_dm_bartlett(loss_diff)
                rows.append({"Model 1": m1, "Model 2": m2, "DM Statistic": dm_stat, "P-Value": p_val})
            dm_by_horizon[h] = pd.DataFrame(rows)
        return dm_by_horizon



                    


        

