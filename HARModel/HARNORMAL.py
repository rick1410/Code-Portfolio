from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.optimize
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1


class HARModel:
    """
    Heterogeneous Autoregressive (HAR) model with Gaussian errors.

    Parameters
    ----------
    realized_volatility : np.ndarray
        1-D array of realized volatilities (or variances).

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Error distribution ('Normal').
    only_kernel : bool
        Marker indicating the model forecasts volatility directly.
    realized_volatility : np.ndarray
        Realized volatility series as a float array.
    optimal_params : Optional[Dict[str, float]]
        Mapping of fitted parameter names to values when available.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (effective sample multiplied by mean ll with sign preserved as in original code).
    aic : Optional[float]
        Akaike Information Criterion.
    bic : Optional[float]
        Bayesian Information Criterion.
    convergence : Optional[bool]
        Optimizer success flag.
    standard_errors : Optional[np.ndarray]
        Approximate standard errors from inverse Hessian; NaN on failure.
    """

    model_name: str = "HAR"
    distribution: str = "Normal"
    only_kernel: bool = True

    realized_volatility: np.ndarray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: Optional[bool]
    standard_errors: Optional[np.ndarray]

    def __init__(self, realized_volatility: np.ndarray) -> None:
        self.realized_volatility = np.asarray(realized_volatility, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = None
        self.standard_errors = None

    def _har_mean_equation(self, t: int, params: Sequence[float]) -> float:
        """HAR mean equation μ_t = α0 + α1*RV_{t-1} + α2*RV^W_{t-1} + α3*RV^M_{t-1}.

        Parameters
        ----------
        t : int
            Current index (t >= 1).
        params : sequence of float
            (alpha0, alpha1, alpha2, alpha3, sigma2).

        Returns
        -------
        float
            μ_t implied by HAR structure.
        """
        alpha0, alpha1, alpha2, alpha3, _ = params
        rv = self.realized_volatility
        daily_lag = rv[t - 1]
        weekly_lag = float(np.mean(rv[max(0, t - 5):t]))
        monthly_lag = float(np.mean(rv[max(0, t - 22):t]))
        return float(alpha0 + alpha1 * daily_lag + alpha2 * weekly_lag + alpha3 * monthly_lag)

    def log_likelihood(self, params: Sequence[float]) -> float:
        """Average negative log-likelihood under Normal errors.

        Parameters
        ----------
        params : sequence of float
            (alpha0, alpha1, alpha2, alpha3, sigma2).

        Returns
        -------
        float
            Average negative log-likelihood (to minimize).
        """
        alpha0, alpha1, alpha2, alpha3, sigma2 = params
        if sigma2 <= 0.0: return 1e10

        rv = self.realized_volatility
        T = len(rv)
        residuals: List[float] = []
        for t in range(22, T):
            mu_t = self._har_mean_equation(t, (alpha0, alpha1, alpha2, alpha3, sigma2))
            residuals.append(rv[t] - mu_t)

        if not residuals: return 1e10
        res = np.asarray(residuals, dtype=float)
        nll_i = 0.5 * np.log(2.0 * np.pi) + 0.5 * np.log(sigma2) + 0.5 * (res**2 / sigma2)
        return float(np.mean(nll_i))

    def compute_aic_bic(self, log_likelihood: float, num_params: int) -> Tuple[float, float]:
        """Compute AIC and BIC (preserving original scaling).

        Parameters
        ----------
        log_likelihood : float
            Total log-likelihood over the effective sample (as used by the original code).
        num_params : int
            Number of parameters.

        Returns
        -------
        (float, float)
            (AIC, BIC).
        """
        T = len(self.realized_volatility)
        aic = 2.0 * num_params - 2.0 * log_likelihood
        bic = np.log(T) * num_params - 2.0 * log_likelihood
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """Estimate parameters by SLSQP subject to bounds.

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum or data-driven defaults.
        compute_metrics : bool, default False
            If True, also compute AIC/BIC, total log-likelihood, and standard errors.

        Returns
        -------
        dict or tuple
            Params dict; or (params, AIC, BIC, total_loglik, SEs) when `compute_metrics=True`.
        """
        rv = self.realized_volatility
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {
                    "alpha0": float(np.mean(rv)) if rv.size > 0 else 0.0,
                    "alpha1": 0.5,
                    "alpha2": 0.2,
                    "alpha3": 0.1,
                    "sigma2": float(np.var(rv) * 0.5) if rv.size > 1 else 1e-12,
                }

        names = list(initial_params.keys())
        x0 = list(initial_params.values())
        bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (1e-12, 1e12)]

        result = scipy.optimize.minimize(self.log_likelihood, x0, method="SLSQP", bounds=bounds)
        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(names, [float(v) for v in result.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")
            return self.optimal_params

        if compute_metrics and self.convergence:
            T_eff = max(0, len(rv) - 22)
            self.log_likelihood_value = float(-result.fun * T_eff)
            k = len(names)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, k)

            opt_pars = np.asarray(list(self.optimal_params.values()), dtype=float)
            hess_mean = approx_hess1(opt_pars, self.log_likelihood)
            hess_full = hess_mean * max(T_eff, 1)

            try:
                cov = inv(hess_full)
                self.standard_errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                self.standard_errors = np.full(k, np.nan, dtype=float)

            if np.isnan(self.standard_errors).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = scipy.optimize.approx_fprime(opt_pars, self.log_likelihood, eps)
                fb_hess = np.outer(grad, grad) * max(T_eff, 1)
                try:
                    cov_fb = inv(fb_hess)
                    self.standard_errors = np.sqrt(np.maximum(np.diag(cov_fb), 0.0))
                except np.linalg.LinAlgError:
                    self.standard_errors = np.full(k, np.nan, dtype=float)

            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors
        return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """HAR-based multi-step forecasts using recursive substitution.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        np.ndarray
            Array of forecasts of length `horizon`.
        """
        if self.optimal_params is None: raise ValueError("Model must be optimized before forecasting.")

        a0 = self.optimal_params["alpha0"]; a1 = self.optimal_params["alpha1"]; a2 = self.optimal_params["alpha2"]; a3 = self.optimal_params["alpha3"]
        rv_hist: List[float] = list(self.realized_volatility.astype(float))
        forecasts: List[float] = []

        for _ in range(int(horizon)):
            daily = rv_hist[-1]
            weekly = float(np.mean(rv_hist[-5:])) if len(rv_hist) >= 5 else float(np.mean(rv_hist))
            monthly = float(np.mean(rv_hist[-22:])) if len(rv_hist) >= 22 else float(np.mean(rv_hist))
            f = float(a0 + a1 * daily + a2 * weekly + a3 * monthly)
            forecasts.append(f)
            rv_hist.append(f)

        return np.asarray(forecasts, dtype=float)
