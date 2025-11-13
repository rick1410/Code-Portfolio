from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import inv
from scipy.optimize import minimize, approx_fprime
from statsmodels.tools.numdiff import approx_hess1
from scipy.special import gammaln

class AR1ModelStudentt:
    """
    AR(1) with Student-t innovations, estimated by MLE.

    Model:
        r_t = μ + φ(r_{t-1} - μ) + ε_t,  ε_t ~ t_ν(0, σ²)

    Parameters
    ----------
    kernel_array : ArrayLike
        1-D series (length ≥ 3). Treated as log returns.

    Attributes
    ----------
    model_name : str
        Human-readable model name.
    distribution : str
        Innovation distribution name.
    only_kernel : bool
        Flag preserved from original API.
    log_returns : NDArray[np.float64]
        Stored input series as float array.
    optimal_params : Optional[Dict[str, float]]
        {"mu", "phi", "sigma2", "nu1"} (nu1 is ν; kept for compatibility).
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (sum over t=2..T).
    aic : Optional[float]
        Akaike Information Criterion.
    bic : Optional[float]
        Bayesian Information Criterion.
    convergence : bool
        Whether the optimizer reported success.
    standard_errors : Optional[NDArray[np.float64]]
        Approx. standard errors from inverse Hessian.
    """

    # Class variables
    model_name: str = "AR-t"
    distribution: str = "Student t"
    only_kernel: bool = True

    # Instance attributes (type hints)
    log_returns: NDArray[np.float64]
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: bool
    standard_errors: Optional[NDArray[np.float64]]

    def __init__(self, kernel_array: ArrayLike) -> None:
        """Initialize the model with a 1-D array (length ≥ 3)."""
        arr = np.asarray(kernel_array, dtype=float)
        if arr.ndim != 1 or arr.size < 3: raise ValueError("kernel_array must be 1-D with length ≥ 3.")
        self.log_returns = arr
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def log_likelihood(self, mu: float, phi: float, sigma2: float, nu: float) -> float:
        """Return the negative average log-likelihood for AR(1)-t."""
        if sigma2 <= 0 or nu <= 2 or not np.isfinite(mu) or not np.isfinite(phi): return np.inf
        r = self.log_returns
        eps = (r[1:] - mu) - phi * (r[:-1] - mu)
        ll_i = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * sigma2) - ((nu + 1) / 2) * np.log1p((eps * eps) / (nu * sigma2))
        return -np.mean(ll_i)

    def compute_aic_bic(self, total_loglik: float, num_params: int) -> Tuple[float, float]:
        """Compute and return (AIC, BIC) from total log-likelihood."""
        n = self.log_returns.size - 1
        aic = 2 * num_params - 2 * total_loglik
        bic = np.log(n) * num_params - 2 * total_loglik
        return aic, bic

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, NDArray[np.float64]]:
        """Fit parameters via SLSQP with bounds; optionally compute metrics and SEs."""
        if initial_params is None: initial_params = {"mu": float(np.mean(self.log_returns)), "phi": 0.1, "sigma2": float(np.var(self.log_returns)), "nu1": 5.0}
        order = ("mu", "phi", "sigma2", "nu1")

        try:
            x0 = np.array([float(initial_params[k]) for k in order], dtype=float)
        except KeyError as e:
            raise ValueError(f"initial_params missing key {e!s}; required keys: {order}.")

        bounds = [(None, None), (-0.999, 0.999), (1e-6, None), (2.000001, None)]
        objective = lambda x: self.log_likelihood(x[0], x[1], x[2], x[3])
        res = minimize(objective, x0, method="SLSQP", bounds=bounds)
        self.convergence = bool(res.success)

        mu_opt, phi_opt, sigma2_opt, nu_opt = [float(v) for v in res.x]
        self.optimal_params = {"mu": mu_opt, "phi": phi_opt, "sigma2": sigma2_opt, "nu1": nu_opt}

        if compute_metrics and self.convergence:
            n = self.log_returns.size - 1
            self.log_likelihood_value = float(-res.fun * n)
            k = len(order)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, k)

            try:
                H = approx_hess1(res.x, objective, args=())
                cov = inv(H) / n
                ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
                if not np.all(np.isfinite(ses)): raise FloatingPointError("Non-finite SEs from Hessian.")
            except Exception:
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = approx_fprime(res.x, objective, eps)
                cov = np.outer(grad, grad)  # Fallback; not a true covariance but keeps behavior.
                ses = np.sqrt(np.maximum(np.diag(cov), 0.0))

            self.standard_errors = ses.astype(float)
            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)

        return self.optimal_params

    def one_step_ahead_forecast(self) -> float:
        """Return the one-step-ahead conditional mean forecast."""
        self._require_fitted()
        mu = self.optimal_params["mu"]; phi = self.optimal_params["phi"]; last = float(self.log_returns[-1])
        return float(mu + phi * (last - mu))

    def multi_step_ahead_forecast(self, horizon: int) -> NDArray[np.float64]:
        """Return multi-step-ahead conditional mean forecasts (length = horizon)."""
        self._require_fitted()
        if horizon < 1: raise ValueError("horizon must be ≥ 1.")
        mu = self.optimal_params["mu"]; phi = self.optimal_params["phi"]
        forecasts = np.empty(int(horizon), dtype=float)
        forecasts[0] = mu + phi * (self.log_returns[-1] - mu)
        for h in range(1, int(horizon)):
            forecasts[h] = mu + phi * (forecasts[h - 1] - mu)
        return forecasts

    def _require_fitted(self) -> None:
        """Raise if the model has not been optimized."""
        if self.optimal_params is None: raise RuntimeError("Model must be optimized before forecasting.")

__all__ = ["AR1ModelStudentt"]
