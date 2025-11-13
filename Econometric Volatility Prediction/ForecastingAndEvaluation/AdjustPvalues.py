from typing import Dict, Any
import pandas as pd
from statsmodels.stats.multitest import multipletests

class AdjustForMultipleTests:
    """
    Adjust p-values for multiple model comparisons per stock and horizon.

    This utility applies a false discovery rate (FDR) procedure (or another
    `statsmodels`-supported method) across models for each stock, test, and
    forecasting horizon. It preserves the original behavior: tests
    considered are "UC", "CC", and "IND", with p-values extracted from
    each model's result vector at indices (2*test_idx + 1).

    Notes
    -----
    Expected structure of `all_results`:
      all_results[ticker]['combined']['backtest_results'][horizon][model] -> list/tuple
    The list/tuple must contain p-values at positions 1, 3, 5 corresponding
    to tests in order ["UC","CC","IND"].

    Attributes
    ----------
    None
        This class is a pure utility holder; it has no instance attributes.
    """

    @staticmethod
    def var_backtest_fdr_by_stock(all_results: Dict[str, Any], alpha: float = 0.05, method: str = 'fdr_bh') -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Apply multiple-testing correction across models for each stock/test/horizon.

        Parameters
        ----------
        all_results : dict
            Nested results dictionary; see Notes for expected structure.
        alpha : float, default 0.05
            Family-wise significance level passed to `multipletests`.
        method : str, default 'fdr_bh'
            Correction method per `statsmodels.stats.multitest.multipletests`.

        Returns
        -------
        dict[str, dict[str, pd.DataFrame]]
            Mapping test -> ticker -> DataFrame of adjusted p-values (index=models, columns=horizons).
        """
        tests = ['UC', 'CC', 'IND']
        adjusted: Dict[str, Dict[str, pd.DataFrame]] = {test: {} for test in tests}

        for ticker, res in all_results.items():
            bt = res['combined']['backtest_results']
            horizons = sorted(bt.keys())

            for test_idx, test in enumerate(tests):
                models = sorted(bt[horizons[0]].keys())
                df_adj = pd.DataFrame(index=models, columns=horizons, dtype=float)

                for h in horizons:
                    raw = {model: float(vals[2 * test_idx + 1]) for model, vals in bt[h].items()}
                    model_list = list(raw.keys())
                    pvals = [raw[m] for m in model_list]
                    _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
                    df_adj.loc[model_list, h] = p_adj

                adjusted[test][ticker] = df_adj

        return adjusted
