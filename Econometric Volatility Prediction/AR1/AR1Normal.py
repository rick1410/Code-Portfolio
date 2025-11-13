import numpy as np
import scipy
from typing import Dict, Optional, Tuple, Union
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.optimize import approx_fprime

class AR1Model:
    """
    AR(1) model with Normal innovations, optimized via maximum likelihood.

    Model:
        r_t = μ + φ (r_{t-1} - μ) + ε_t,   ε_t ~ N(0, σ²)

    Parameters
    ----------
    kernel_array : array-like
        The observed time series (e.g., log-returns). Will be converted to a NumPy array.

    Attributes
    ----------
    model_name : str
        Short name of the model.
    distribution : str
        Innovation distribution name.
    only_kernel : bool
        Indicator for pipelines that use only the kernel/series as input.
    log_returns : np.ndarray
        Stored series as a 1-D NumPy array.
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters after optimization: {"mu": μ, "phi": φ, "sigma2": σ²}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at the optimum (sum over t).
    aic : Optional[float]
        Akaike Information Criterion at the optimum.
    bic : Optional[float]
        Bayesian Information Criterion at the optimum.
    convergence : bool
        Whether the optimizer reported success.
    standard_errors : Optional[np.ndarray]
        Approximate standard errors for parameters (sqrt of diagonal of covariance).
    """

    # Class variables (typed)
    model_name: str = "AR"
    distribution: str = "Normal"
    only_kernel: bool = True

    # Instance attributes (type hints for clarity)
    log_returns: np.ndarray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: bool
    standard_errors: Optional[np.ndarray]

    def __init__(self, kernel_array: Union[np.ndarray, list, tuple]):
        """Initialize the AR(1) model with a time series."""
        self.log_returns = np.asarray(kernel_array, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def log_likelihood(self, params: Union[np.ndarray, Tuple[float, float, float]]) -> float:
        """Return the negative average log-likelihood for the AR(1)-Normal model."""
        mu, phi, sigma2 = params
        r = self.log_returns
        residuals = r[1:] - mu - phi * (r[:-1] - mu)
        ll = -0.5 * (np.log(2 * np.pi * sigma2) + (residuals ** 2) / sigma2)
        return -np.mean(ll)

    def compute_aic_bic(self, total_loglik: float, num_params: int) -> Tuple[float, float]:
        """Compute and return (AIC, BIC)."""
        n = len(self.log_returns)
        aic = 2 * num_params - 2 * total_loglik
        bic = np.log(n) * num_params - 2 * total_loglik
        return aic, bic

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """Fit the AR(1) model by minimizing the negative average log-likelihood."""
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                mu0 = float(np.mean(self.log_returns))
                phi0 = 0.1
                sigma20 = float(np.var(self.log_returns))
                initial_params = {"mu": mu0, "phi": phi0, "sigma2": sigma20}

        keys = list(initial_params.keys())
        x0 = np.array(list(initial_params.values()), dtype=float)

        bounds = [(None, None), (-0.999, 0.999), (1e-6, None)]
        result = scipy.optimize.minimize(self.log_likelihood, x0, method="SLSQP", bounds=bounds)

        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, result.x))
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")

        if compute_metrics and self.convergence:
            n = len(self.log_returns)
            self.log_likelihood_value = -float(result.fun) * n
            num_params = len(x0)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, num_params)

            H = approx_hess1(np.array(list(self.optimal_params.values()), dtype=float), self.log_likelihood, args=())
            cov = inv(H) / n
            ses = np.sqrt(np.diag(cov))
            if np.isnan(ses).any():
                eps = np.sqrt(np.finfo(float).eps)
                grad = approx_fprime(np.array(list(self.optimal_params.values()), dtype=float), self.log_likelihood, eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.diag(cov_alt))
            self.standard_errors = ses

            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)
        return self.optimal_params

    def one_step_ahead_forecast(self) -> float:
        """Compute and return the one-step-ahead conditional mean forecast."""
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")
        mu = self.optimal_params["mu"]
        phi = self.optimal_params["phi"]
        return float(mu + phi * (self.log_returns[-1] - mu))

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """Compute and return multi-step-ahead conditional mean forecasts."""
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")
        mu = self.optimal_params["mu"]
        phi = self.optimal_params["phi"]
        last = self.log_returns[-1]
        forecasts = np.empty(horizon, dtype=float)
        forecasts[0] = mu + phi * (last - mu)
        for h in range(1, horizon):
            forecasts[h] = mu + phi * (forecasts[h - 1] - mu)
        return forecasts
