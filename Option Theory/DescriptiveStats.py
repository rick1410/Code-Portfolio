from typing import Tuple
import numpy as np
import pandas as pd
from scipy import stats

class ReturnStats:
    """Basic stats and hypothesis test utilities on monthly returns."""
    def __init__(self, monthly_df: pd.DataFrame, price_col: str):
        self.df = monthly_df
        self.price_col = price_col

    @property
    def net_returns(self) -> pd.Series:
        return self.df["NetReturn"]

    @property
    def log_returns(self) -> pd.Series:
        return self.df["LogReturn"]

    def mean_std(self) -> Tuple[float, float]:
        mu = float(self.net_returns.mean())
        sigma = float(self.net_returns.std(ddof=1))
        return mu, sigma

    def ttest_mean_zero(self) -> Tuple[float, float]:
        t_stat, p_val = stats.ttest_1samp(self.net_returns, 0.0, alternative="two-sided")
        return float(t_stat), float(p_val)
