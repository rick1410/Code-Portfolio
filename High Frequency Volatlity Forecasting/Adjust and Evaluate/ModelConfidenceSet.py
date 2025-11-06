from typing import Dict, Any, Iterator, Sequence, Tuple
import numpy as np
import pandas as pd

def block_bootstrap(n: int, n_boot: int, block_len: int) -> Iterator[np.ndarray]:
    """
    Circular block bootstrap index generator.

    Parameters
    ----------
    n : int
        Sample length to resample.
    n_boot : int
        Number of bootstrap replicates to generate.
    block_len : int
        Block length for the circular block bootstrap.

    Yields
    ------
    np.ndarray
        Array of indices of length `n` representing a bootstrap resample.
    """
    for _ in range(n_boot):
        indices = np.zeros(n, dtype=np.int32)
        indices[0] = int(np.random.random() * n)
        counter = 0
        for t in range(1, n):
            counter = (counter + 1) % block_len
            if counter == 0:
                indices[t] = int(np.random.random() * n)
            else:
                indices[t] = indices[t - 1] + 1
        indices[indices > n - 1] -= n
        yield indices

def pval_R(z: np.ndarray, z_data: np.ndarray) -> float:
    """
    Max-statistic bootstrap p-value for Model Confidence Set (MCS).

    Parameters
    ----------
    z : np.ndarray
        Bootstrapped standardized loss differences with shape (B, m, m).
    z_data : np.ndarray
        Observed standardized loss differences with shape (m, m).

    Returns
    -------
    float
        Two-sided p-value based on the max absolute statistic.
    """
    TR_dist = np.abs(z).max(axis=(1, 2))
    TR = np.abs(z_data).max()
    return float((TR_dist > TR).mean())

class ModelConfidenceSet:
    """
    Model Confidence Set (MCS) procedure using block bootstrap.

    Parameters
    ----------
    losses : pd.DataFrame
        T x M matrix of loss realizations (columns are models).
    n_boot : int
        Number of bootstrap replicates.
    alpha : float
        Target confidence level (e.g., 0.05).
    block_len : int
        Bootstrap block length.

    Attributes
    ----------
    model_names : pd.Index
        Names of models (DataFrame columns).
    losses : np.ndarray
        Loss matrix as ndarray.
    n_boot : int
        Number of bootstrap replicates.
    alpha : float
        Target confidence level.
    block_len : int
        Bootstrap block length.
    included : np.ndarray | None
        Indices (1-based) of models included in the final MCS.
    excluded : np.ndarray | None
        Indices (1-based) of models excluded from the MCS.
    pvalues : np.ndarray | None
        Sequential MCS p-values aligned with exclusion order.
    """

    model_names: pd.Index
    losses: np.ndarray
    n_boot: int
    alpha: float
    block_len: int
    included: np.ndarray | None
    excluded: np.ndarray | None
    pvalues: np.ndarray | None

    def __init__(self, losses: pd.DataFrame, n_boot: int, alpha: float, block_len: int) -> None:
        self.model_names = losses.columns
        self.losses = losses.values
        self.n_boot = n_boot
        self.alpha = alpha
        self.block_len = block_len
        self.included = None
        self.excluded = None
        self.pvalues = None

    def run(self) -> None:
        """
        Execute the MCS elimination procedure and store results in attributes.
        """
        n, m_start = self.losses.shape
        mloss = self.losses.mean(axis=0, keepdims=True)
        dij_bar = mloss - mloss.T

        dij_bar_bootstrap = np.zeros((self.n_boot, m_start, m_start))
        for i, idx in enumerate(block_bootstrap(n, self.n_boot, self.block_len)):
            bootstrap_loss = self.losses[idx, :].mean(axis=0, keepdims=True)
            dij_bar_bootstrap[i, ...] = bootstrap_loss - bootstrap_loss.T

        dij_std = np.sqrt(np.mean((dij_bar_bootstrap - dij_bar) ** 2, axis=0) + np.eye(m_start))
        z_start = (dij_bar_bootstrap - dij_bar) / dij_std
        z_data = dij_bar / dij_std

        excluded = np.zeros(m_start)
        pvals = np.ones(m_start)
        models = np.arange(1, m_start + 1)

        for i in range(m_start - 1):
            included = np.setdiff1d(models, excluded) - 1
            m = len(included)
            scale = m / (m - 1)
            pvals[i] = pval_R(z_start[:, *np.ix_(included, included)], z_data[np.ix_(included, included)])
            di_bar = np.mean(dij_bar[np.ix_(included, included)], axis=0) * scale
            di_bar_bootstrap = (dij_bar_bootstrap[:, *np.ix_(included, included)].mean(axis=1) * scale)
            di_std = np.sqrt(np.mean((di_bar_bootstrap - di_bar) ** 2, axis=0))
            t = di_bar / di_std
            excluded[i] = included[np.argmax(t)] + 1

        pvals = np.maximum.accumulate(pvals)
        excluded[-1] = np.setdiff1d(models, excluded)[-1]

        self.included = excluded[pvals >= self.alpha]
        self.excluded = excluded[pvals < self.alpha]
        self.pvalues = pvals

    def results(self) -> pd.DataFrame:
        """
        Run MCS and return a summary DataFrame.

        Returns
        -------
        pd.DataFrame
            Index is model names ordered by exclusion then inclusion; columns:
            - 'pvalues': sequential MCS p-values
            - 'status' : 'included' or 'excluded'
        """
        self.run()
        index = np.concatenate([self.excluded, self.included]).astype(int) - 1
        results_dict = { "pvalues": self.pvalues,"status": np.where(self.pvalues >= self.alpha, "included", "excluded"),"models": self.model_names[index]}
        df = pd.DataFrame(results_dict)
        df.index = df.pop("models")
        return df

def compute_mcs(combined: Dict[str, Any], horizons: Sequence[int], alpha: float = 0.05, n_boot: int = 2000) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Compute Model Confidence Sets for a single asset across multiple horizons.

    Parameters
    ----------
    combined : dict
        Dictionary with keys:
          - "forecasts": dict[h] -> dict[model] -> array-like forecasts
          - "realized_kernel": dict[h] -> array-like realizations
    horizons : sequence of int
        Horizons to evaluate.
    alpha : float, default 0.05
        Target confidence level.
    n_boot : int, default 2000
        Number of bootstrap replicates.

    Returns
    -------
    dict[int, dict[str, pd.DataFrame]]
        Mapping horizon -> {"mcs": DataFrame}.
    """
    mcs_results: Dict[int, Dict[str, pd.DataFrame]] = {}
    for h in horizons:
        model_names = list(combined["forecasts"][h].keys())
        y_true = np.array(combined["realized_kernel"][h])
        loss_matrix = np.column_stack([(y_true - np.array(combined["forecasts"][h][model])) ** 2 for model in model_names])
        loss_df = pd.DataFrame(loss_matrix, columns=model_names)
        mcs = ModelConfidenceSet(losses=loss_df, n_boot=n_boot, alpha=alpha, block_len=int(np.sqrt(loss_matrix.shape[0])))
        mcs_df = mcs.results()
        mcs_results[h] = {"mcs": mcs_df}
    return mcs_results
