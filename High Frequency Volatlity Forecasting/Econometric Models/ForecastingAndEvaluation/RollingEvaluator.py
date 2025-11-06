from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import inspect

from ForecastingAndEvaluation.FactorAndDmTest import ForecastUtils
from ForecastingAndEvaluation.rollingwindow import rolling_window
from DataAndRealizedKernel.PDFmodels import PDFModels
from ForecastingAndEvaluation.ValueAtRisk import ValueAtRisk

class RollingEvaluator:
    """
    Rolling-window evaluator for volatility models with VaR backtesting.

    Performs:
      1) One-time optimization on the first window for all models.
      2) Rolling re-estimation (warm-started by previous params) and forecasting
         for multiple horizons.
      3) Risk evaluation (VaR) and backtesting per horizon.

    Parameters
    ----------
    log_returns_array : np.ndarray
        1-D array of log returns.
    kernel_array : np.ndarray
        1-D array of realized-kernel (or volatility target) values.
    inflation_series : np.ndarray
        Series used by inflation-aware models (passed through if requested).
    inflation_dates : Sequence[Any]
        Date index corresponding to `inflation_series`.
    window_index : int
        Rolling window length for training.
    model_classes : Sequence[type]
        Iterable of model classes to be evaluated. Each class must expose
        `model_name` and implement `optimize(...)` and `multi_step_ahead_forecast(h)`.
    confidence_level : float
        VaR confidence level (e.g., 0.99 means 1% tail).
    date_index : Sequence[Any]
        Date index for `log_returns_array`/`kernel_array`.
    K : int
        Additional hyperparameter forwarded to certain models (e.g., inflation-aware).
    horizons : Sequence[int]
        Forecast horizons (positive integers).

    Attributes
    ----------
    log_returns_array : np.ndarray
        Stored returns.
    kernel_array : np.ndarray
        Stored target kernel.
    window_index : int
        Rolling window length.
    horizons : Sequence[int]
        Horizons evaluated.
    confidence_level : float
        VaR confidence level.
    K : int
        Extra hyperparameter forwarded to models when required.
    model_classes : Sequence[type]
        Model classes to evaluate.
    date_index : Sequence[Any]
        Date index.
    inflation_series : np.ndarray
        Inflation series passed to compatible models.
    inflation_dates : Sequence[Any]
        Dates for inflation series.
    rw : Any
        Rolling window splitter from `rolling_window`.
    pdf_model : PDFModels
        PDF model provider for VaR.
    risk_metrics : ValueAtRisk
        VaR calculator/backtester.
    """

    log_returns_array: np.ndarray
    kernel_array: np.ndarray
    window_index: int
    horizons: Sequence[int]
    confidence_level: float
    K: int
    model_classes: Sequence[type]
    date_index: Sequence[Any]
    inflation_series: np.ndarray
    inflation_dates: Sequence[Any]
    rw: Any
    pdf_model: PDFModels
    risk_metrics: ValueAtRisk

    def __init__(self, log_returns_array: np.ndarray, kernel_array: np.ndarray, inflation_series: np.ndarray, inflation_dates: Sequence[Any], window_index: int, model_classes: Sequence[type], confidence_level: float, date_index: Sequence[Any], K: int, horizons: Sequence[int]) -> None:
        self.log_returns_array = log_returns_array
        self.kernel_array = kernel_array
        self.window_index = window_index
        self.horizons = horizons
        self.confidence_level = confidence_level
        self.K = K
        self.model_classes = model_classes
        self.date_index = date_index
        self.inflation_series = inflation_series
        self.inflation_dates = inflation_dates
        self.rw = rolling_window(window=window_index, horizon=max(horizons), period=1)
        self.pdf_model = PDFModels()
        self.risk_metrics = ValueAtRisk(self.pdf_model)


    def _print_rmse_per_model(self, metrics: Dict[int, Dict[str, Dict[str, float]]]) -> None:
        horizons = sorted(metrics.keys())
        models = sorted({m for h in horizons for m in metrics[h].keys()})
        # header
        header = ["Model"] + [f"h={h}" for h in horizons]
        widths = [max(12, len("Model"))] + [max(8, len(f"h={h}")) for h in horizons]
        fmt_cells = lambda row: "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
        print("\n=== RMSE per model per horizon ===")
        print(fmt_cells(header))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        # rows
        for model in models:
            row = [model]
            for h in horizons:
                val = metrics[h].get(model, {}).get("rmse", None)
                row.append(f"{val:.6f}" if val is not None else "—")
            print(fmt_cells(row))
        print("")  # trailing newline

    def optimize_first_window(self) -> Dict[str, Dict[str, Any]]:
        """
        Optimize all models on the first rolling window and collect diagnostics.

        Returns
        -------
        dict
            Mapping model_name -> details (params / metrics / diagnostics).
        """
        splits = self.rw.split(self.log_returns_array)
        first_split = splits[0]
        train_indices, _ = first_split
        train_log_returns = self.log_returns_array[train_indices]
        train_kernel = self.kernel_array[train_indices]
        first_window_results: Dict[str, Dict[str, Any]] = {}

        for model_class in self.model_classes:
            init_params = inspect.signature(model_class.__init__).parameters
            if "X" in init_params or "realized_kernel" in init_params or 'x' in init_params:
                model = model_class(train_log_returns, train_kernel)
            elif "inflation_series" in init_params:
                model = model_class(train_log_returns, self.inflation_series, self.inflation_dates, self.date_index, self.K)
            elif hasattr(model_class, "only_kernel"):
                model = model_class(train_kernel)
            else:
                model = model_class(train_log_returns)

            opt_params = model.optimize(initial_params=None, compute_metrics=True)
            param_dict = opt_params[0] if isinstance(opt_params, tuple) else opt_params

            if hasattr(model, "bayesian"):
                first_window_results[model_class.model_name] = {
                    "params": opt_params[0],
                    "mcmcparams": opt_params[1],
                    "confidence_intervals": getattr(model, "confidence_intervals"),
                    "acceptance_rate": getattr(model, "acceptance_rate"),
                    "confidence_intervals": getattr(model, "confidence_intervals"),
                    "candidate_correlations": getattr(model, "candidate_correlations"),
                }
            else:
                first_window_results[model_class.model_name] = {
                    "params": param_dict,
                    "aic": getattr(model, "aic"),
                    "bic": getattr(model, "bic"),
                    "log_likelihood": getattr(model, "log_likelihood_value"),
                    "std_errs": getattr(model, "standard_errors"),
                    "distribution": getattr(model, "distribution"),
                }

        return first_window_results

    def rolling_evaluation(self) -> Dict[str, Any]:
        """
        Run rolling evaluation: refit/forecast, compute metrics, and backtests.

        Returns
        -------
        dict
            Dictionary with keys:
              - "first_window_results"
              - "metrics"
              - "backtest_results"
              - "forecasts"
              - "realized_kernel"
              - "test_log_returns"
              - "var_results"
              - "hit_var_results"
              - "test_dates"
        """
        splits = self.rw.split(self.log_returns_array)
        first_window_results = self.optimize_first_window()
        previous_params = {mn: first_window_results[mn]["params"] for mn in first_window_results.keys()}

        forecasts: Dict[int, Dict[str, List[float]]] = {h: {m.model_name: [] for m in self.model_classes} for h in self.horizons}
        var_results: Dict[int, Dict[str, List[float]]] = {h: {m.model_name: [] for m in self.model_classes} for h in self.horizons}
        hit_var_results: Dict[int, Dict[str, List[int]]] = {h: {m.model_name: [] for m in self.model_classes} for h in self.horizons}

        realized_kernel: Dict[int, List[float]] = {h: [] for h in self.horizons}
        test_log_returns: Dict[int, List[float]] = {h: [] for h in self.horizons}
        test_dates: List[Any] = []

        for train_indices, test_indices in splits[1:]:
            train_log_returns = self.log_returns_array[train_indices]
            train_kernel = self.kernel_array[train_indices]
            factor = ForecastUtils.calculate_factor(train_log_returns, train_kernel)
            test_dates.append(self.date_index[test_indices[0]])

            for h in self.horizons:
                if h <= len(test_indices):
                    test_log_return_h = self.log_returns_array[test_indices[h - 1]]
                    test_kernel_h = self.kernel_array[test_indices[h - 1]] * factor
                    realized_kernel[h].append(test_kernel_h)
                    test_log_returns[h].append(test_log_return_h)

            for model_class in self.model_classes:
                model_name = model_class.model_name
                init_params_signature = inspect.signature(model_class.__init__).parameters

                if "X" in init_params_signature or "realized_kernel" in init_params_signature or 'x' in init_params_signature:
                    model = model_class(train_log_returns, train_kernel)
                elif "inflation_series" in init_params_signature:
                    model = model_class(train_log_returns, self.inflation_series, self.inflation_dates, self.date_index, self.K)
                elif hasattr(model_class, "only_kernel"):
                    model = model_class(train_kernel)
                else:
                    model = model_class(train_log_returns)

                prev_params = previous_params.get(model_name, None)
                new_params = model.optimize(initial_params=prev_params, compute_metrics=False)

                if model.convergence:
                    previous_params[model_name] = new_params
                else:
                    model.optimal_params = prev_params
                    previous_params[model_name] = prev_params

                for h in self.horizons:
                    if h <= len(test_indices):
                        forecast = model.multi_step_ahead_forecast(h)[-1]
                        forecasts[h][model_name].append(forecast)
                        var = self.risk_metrics.calculate_var(model, forecast, self.confidence_level)
                        var_results[h][model_name].append(var)
                        hit_var_results[h][model_name].append(1 if test_log_return_h < var else 0)

        metrics: Dict[int, Dict[str, Dict[str, float]]] = {}
        backtest_eval: Dict[int, Dict[str, Any]] = {}
        for h in self.horizons:
            metrics[h] = {model_name: {"rmse": float(np.sqrt(mean_squared_error(realized_kernel[h], forecasts[h][model_name]))),"mae": float(mean_absolute_error(realized_kernel[h], forecasts[h][model_name]))} for model_name in forecasts[h]}
            backtest_eval[h] = {model_name: self.risk_metrics.backtest_var(hit_var_results[h][model_name], 1 - self.confidence_level) for model_name in hit_var_results[h]}

        self._print_rmse_per_model(metrics)

        return {"first_window_results": first_window_results,"metrics": metrics,"backtest_results": backtest_eval,"forecasts": forecasts,"realized_kernel": realized_kernel,"test_log_returns": test_log_returns,"var_results": var_results,"hit_var_results": hit_var_results,"test_dates": test_dates,}
