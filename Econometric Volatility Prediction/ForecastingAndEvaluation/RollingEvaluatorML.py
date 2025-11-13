from typing import Any, Dict, List, Sequence, Tuple, Set
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

from .rollingwindow import rolling_window
from .ValueAtRisk import ValueAtRisk
from .FactorAndDmTest import ForecastUtils
from DataAndRealizedKernel.PDFmodels import PDFModels
from FeatureCreation.Features import vol_features


class RollingMLEvaluator:
    """
    Rolling-window evaluator for classical ML regressors with multi-horizon targets.

    Steps:
      1) Build volatility features/targets (`vol_features`), scale X.
      2) Tune each base estimator on the first window via `GridSearchCV`
         (time-series split), wrapping as `MultiOutputRegressor`.
      3) Walk forward: refit/forecast at each window, compute VaR and hits.
      4) Report RMSE/MAE per horizon and VaR backtests.

    Parameters
    ----------
    lookback : int
        Maximum lag order for features.
    ma_windows : Sequence[int]
        Rolling window sizes for MA features.
    log_returns : np.ndarray
        1-D array of log returns.
    kernel : np.ndarray
        1-D realized-kernel series (target base).
    sentiment : np.ndarray
        1-D categorical sentiment array aligned with returns.
    window : int
        Rolling window length (in feature rows).
    period : int
        Step between windows (stride).
    models : Sequence[Tuple[Any, Dict[str, Any]]]
        Sequence of (base_estimator, param_grid) pairs.
    confidence_level : float
        VaR confidence (e.g., 0.99 => 1% tail).
    date_index : Sequence[Any]
        Dates aligned with raw series, used for reporting.
    horizons : Sequence[int]
        Forecast horizons (positive integers).

    Attributes
    ----------
    log_ret_raw, kernel_raw, sentiment : np.ndarray
        Stored raw series.
    CL : float
        VaR confidence level.
    alpha : float
        1 - CL (used in backtesting).
    horizons : Sequence[int]
        Horizons evaluated.
    inner_cv : int
        Number of CV splits for first-window tuning.
    n_add : int
        Kept for parity with prior API (not used directly here).
    risk : ValueAtRisk
        VaR calculator.
    scaler : StandardScaler
        Placeholder scaler attribute (kept for API parity).
    X, Y : np.ndarray
        Engineered features and targets (multi-output).
    offset : int
        Number of initial raw rows dropped due to feature engineering.
    rw : Any
        Rolling window splitter.
    models : Sequence[Tuple[Any, Dict[str, Any]]]
        Model specs received in constructor.
    best_params : Dict[str, Dict[str, Any]]
        Best param grid results per model name.
    estimators : Dict[str, MultiOutputRegressor]
        Tuned, wrapped estimators.
    _forest_growable : Set[str]
        Names of models supporting warm_start + n_estimators.
    forecasts, VaR, hits : dict
        Rolling outputs per horizon and model.
    realised_kernel_adj, test_returns : dict
        Factor-adjusted realized kernel and realized returns.
    date_index, test_dates : list-like
        Dates aligned to test steps.
    """

    # Core config
    log_ret_raw: np.ndarray
    kernel_raw: np.ndarray
    sentiment: np.ndarray
    CL: float
    alpha: float
    horizons: Sequence[int]
    inner_cv: int
    n_add: int
    risk: ValueAtRisk
    scaler: StandardScaler

    # Features / targets
    X: np.ndarray
    Y: np.ndarray
    offset: int

    # Rolling infra
    rw: Any
    models: Sequence[Tuple[Any, Dict[str, Any]]]

    # Fitted estimators and params
    best_params: Dict[str, Dict[str, Any]]
    estimators: Dict[str, MultiOutputRegressor]
    _forest_growable: Set[str]

    # Outputs
    forecasts: Dict[int, Dict[str, List[float]]]
    VaR: Dict[int, Dict[str, List[float]]]
    hits: Dict[int, Dict[str, List[int]]]
    realised_kernel_adj: Dict[int, List[float]]
    test_returns: Dict[int, List[float]]
    date_index: Sequence[Any]
    test_dates: List[Any]

    def __init__(self, lookback: int, ma_windows: Sequence[int], log_returns: np.ndarray, kernel: np.ndarray, sentiment: np.ndarray, window: int, period: int, models: Sequence[Tuple[Any, Dict[str, Any]]], confidence_level: float, date_index: Sequence[Any], horizons: Sequence[int]) -> None:
        self.log_ret_raw = log_returns
        self.kernel_raw = kernel
        self.sentiment = sentiment
        self.CL = confidence_level
        self.alpha = 1.0 - confidence_level
        self.horizons = horizons
        self.inner_cv = 2
        self.n_add = 10
        self.risk = ValueAtRisk(PDFModels())
        self.scaler = StandardScaler()

        self.X, self.Y = vol_features(self.log_ret_raw, self.kernel_raw, self.sentiment, lookback, ma_windows, horizons=self.horizons)
        X_scaler = StandardScaler().fit(self.X)
        self.X = X_scaler.transform(self.X)
        self.offset = len(self.log_ret_raw) - len(self.Y)

        self.rw = rolling_window(window, max(horizons), period)
        self.models = models

        self.best_params = {}
        self.estimators = {}
        self._forest_growable = set()

        self.forecasts = {h: defaultdict(list) for h in self.horizons}
        self.VaR = {h: defaultdict(list) for h in self.horizons}
        self.hits = {h: defaultdict(list) for h in self.horizons}
        self.realised_kernel_adj = {h: [] for h in self.horizons}
        self.test_returns = {h: [] for h in self.horizons}
        self.date_index = date_index
        self.test_dates = []

    @staticmethod
    def _is_growable(est: Any) -> bool:
        """
        Whether estimator can grow trees via warm start.

        Parameters
        ----------
        est : Any
            Estimator (possibly wrapped) to check.

        Returns
        -------
        bool
            True if it supports `warm_start` and `n_estimators`.
        """
        return hasattr(est, "warm_start") and hasattr(est, "n_estimators")

    def _tune_first_window(self, tr_idx: np.ndarray) -> None:
        """
        Tune base estimators on the first rolling window via grid search.

        Parameters
        ----------
        tr_idx : np.ndarray
            Indices (in feature space) for the first training window.
        """
        X0, y0 = self.X[tr_idx], self.Y[tr_idx]
        for base_est, grid in self.models:
            wrapped = MultiOutputRegressor(base_est)
            wrapped_grid = {f"estimator__{k}": v for k, v in grid.items()}
            gs = GridSearchCV(estimator=wrapped, param_grid=wrapped_grid, cv=TimeSeriesSplit(n_splits=self.inner_cv), scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(X0, y0)

            best = deepcopy(gs.best_estimator_)
            model_name = best.estimator.__class__.__name__
            if self._is_growable(best):
                best.set_params(warm_start=True)
                self._forest_growable.add(model_name)

            self.best_params[model_name] = gs.best_params_
            self.estimators[model_name] = best

    def run(self) -> Dict[str, Any]:
        """
        Execute rolling evaluation with tuned estimators.

        Returns
        -------
        dict
            Dictionary containing forecasts, VaR, hits, metrics, backtests, and metadata.
        """
        splits = list(self.rw.split(self.Y))
        train0, _ = splits[0]
        self._tune_first_window(np.asarray(train0))

        for i, (train_index, test_index) in enumerate(splits):
            print(f"[{i+1}/{len(splits)}] Running window {i+1}...", flush=True)
            train_index = np.asarray(train_index, dtype=int)
            test_index = np.asarray(test_index, dtype=int)

            X_train, Y_train = self.X[train_index], self.Y[train_index]
            X_test = self.X[test_index]
            train_raw = train_index + self.offset
            test_raw = test_index + self.offset

            factor = ForecastUtils.calculate_factor(self.log_ret_raw[train_raw], self.kernel_raw[train_raw])
            self.test_dates.append(self.date_index[test_raw[0]])

            for h_idx, h in enumerate(self.horizons):
                if h <= len(test_raw):
                    rk_adj = float(self.kernel_raw[test_raw[h - 1]] * factor)
                    self.realised_kernel_adj[h].append(rk_adj)
                    self.test_returns[h].append(float(self.log_ret_raw[test_raw[h - 1]]))

            for name, est in self.estimators.items():
                est.fit(X_train, Y_train)
                y_pred = est.predict(X_test)[0]
                for h_idx, h in enumerate(self.horizons):
                    if h_idx < len(y_pred) and test_raw[h - 1] < len(self.kernel_raw):
                        var_hat = float(y_pred[h_idx])
                        self.forecasts[h][name].append(var_hat)
                        var_next = float(self.risk.calculate_var_ML(var_hat, self.CL, self.log_ret_raw[train_raw], self.kernel_raw[train_raw]))
                        self.VaR[h][name].append(var_next)
                        self.hits[h][name].append(int(self.test_returns[h][-1] < var_next))

        metrics = {h: {model_name: {"rmse": float(np.sqrt(mean_squared_error(self.realised_kernel_adj[h], self.forecasts[h][model_name]))), "mae": float(mean_absolute_error(self.realised_kernel_adj[h], self.forecasts[h][model_name]))} for model_name in self.forecasts[h]} for h in self.horizons}
        backtest_results = {h: {model_name: ValueAtRisk.backtest_var(self.hits[h][model_name], self.alpha) for model_name in self.hits[h]} for h in self.horizons}

        return {
            "first_window_results": self.best_params,
            "forecasts": self.forecasts,
            "var_results": self.VaR,
            "hit_var_results": self.hits,
            "metrics": metrics,
            "backtest_results": backtest_results,
            "realized_kernel": self.realised_kernel_adj,
            "test_log_returns": self.test_returns,
            "test_dates": self.test_dates,
        }
