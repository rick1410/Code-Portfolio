from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.optimize import approx_fprime
from scipy.special import gamma

class AR1astModel:
    """
    AR(1) model with Asymmetric Student-t (AST) innovations, estimated by MLE.

    Model
    -----
    y_t = mu + phi (y_{t-1} - μ) + ε_t,  ε_t ~ AST(delta, ν1, ν2)

    Parameters
    ----------
    kernel_array : ArrayLike
        1-D time series (treated as log returns).

    Attributes
    ----------
    model_name : str
        Short name of the model.
    distribution : str
        Innovation distribution name ("AST").
    only_kernel : bool
        Indicator that this model uses only the kernel/series.
    kernel : NDArray[np.float64]
        Stored input series as a float array.
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters: {"mu", "phi", "delta", "v1", "v2"}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at the optimum (sum over t=2..T).
    aic : Optional[float]
        Akaike Information Criterion at the optimum.
    bic : Optional[float]
        Bayesian Information Criterion at the optimum.
    convergence : Optional[bool]
        Whether the optimizer reported success.
    standard_errors : Optional[NDArray[np.float64]]
        Approximate standard errors for parameters (from inverse Hessian or fallback).
    """

    model_name: str = "AR-AST"
    distribution: str = "AST"
    only_kernel: bool = True

    kernel: NDArray[np.float64]
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: Optional[bool]
    standard_errors: Optional[NDArray[np.float64]]

    def __init__(self, kernel_array: ArrayLike) -> None:
        """Initialize the AR(1)-AST model with a time series."""
        self.kernel = np.asarray(kernel_array, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = None
        self.standard_errors = None

    def K(self, v: float) -> float:
        """Return K(v) = Γ((v+1)/2) / (√(π v) Γ(v/2))."""
        return float(gamma((v + 1) / 2) / (np.sqrt(np.pi * v) * gamma(v / 2)))

    def B(self, delta: float, v1: float, v2: float) -> float:
        """Return B(delta, ν1, ν2) = delta K(ν1) + (1-delta) K(ν2)."""
        return float(delta * self.K(v1) + (1 - delta) * self.K(v2))

    def alpha_star(self, delta: float, v1: float, v2: float) -> float:
        """Return alpha* = delta K(ν1) / B(delta, ν1, ν2)."""
        B = self.B(delta, v1, v2)
        return float((delta * self.K(v1)) / B)

    def m(self, delta: float, v1: float, v2: float) -> float:
        """Return m(delta, ν1, ν2) used in AST location adjustment."""
        a_star = self.alpha_star(delta, v1, v2)
        B = self.B(delta, v1, v2)
        return float(4 * B * (-(a_star**2) * v1 / (v1 - 1) + (1 - a_star) ** 2 * v2 / (v2 - 1)))

    def s(self, delta: float, v1: float, v2: float) -> float:
        """Return s(delta, ν1, ν2) used in AST scale adjustment."""
        a_star = self.alpha_star(delta, v1, v2)
        B = self.B(delta, v1, v2)
        m = self.m(delta, v1, v2)
        return float(np.sqrt(4 * (delta * a_star**2 * v1 / (v1 - 2) + (1 - delta) * (1 - a_star) ** 2 * v2 / (v2 - 2)) - m**2))

    def I_t(self, eps: float, mu_t: float, h_t: float, m: float, s: float) -> int:
        """Indicator I_t = 1{ m + s (eps - mu_t)/sqrt(h_t) > 0 }."""
        return 1 if (m + s * (eps - mu_t) / np.sqrt(h_t)) > 0 else 0

    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Negative average log-likelihood for AR(1) with AST innovations.

        Parameters
        ----------
        params : array-like of shape (5,)
            Parameter vector [mu, phi, delta, v1, v2].

        Returns
        -------
        float
            Negative average log-likelihood (lower is better).
        """
        mu, phi, delta, v1, v2 = params
        y = self.kernel
        n = len(y) - 1
        eps = y[1:] - mu - phi * (y[:-1] - mu)

        m = self.m(delta, v1, v2)
        s = self.s(delta, v1, v2)
        a_star = self.alpha_star(delta, v1, v2)
        B = self.B(delta, v1, v2)

        ll = np.zeros(n)
        for t in range(n):
            mu_t = mu + phi * (y[t] - mu)
            h_t = 1.0
            I = self.I_t(eps[t], mu, h_t, m, s)
            term1 = (v1 + 1) / 2 * np.log(1 + (1 / v1) * ((m + s * eps[t]) / (2 * a_star)) ** 2)
            term2 = (v2 + 1) / 2 * np.log(1 + (1 / v2) * ((m + s * eps[t]) / (2 * (1 - a_star))) ** 2)
            ll[t] = np.log(s) + np.log(B) - (1 - I) * term1 - I * term2
        return -np.mean(ll)

    def compute_aic_bic(self, total_loglik: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC from total log-likelihood.

        Parameters
        ----------
        total_loglik : float
            Total (summed) log-likelihood at the optimum.
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        tuple[float, float]
            (AIC, BIC).
        """
        n = len(self.kernel) - 1
        aic = 2 * num_params - 2 * total_loglik
        bic = np.log(n) * num_params - 2 * total_loglik
        return aic, bic

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, NDArray[np.float64]]:
        """
        Fit parameters via SLSQP with bounds; optionally compute metrics and SEs.

        Parameters
        ----------
        initial_params : dict or None
            Initial guess {"mu","phi","delta","v1","v2"}. If None, use data-driven defaults.
        compute_metrics : bool, default False
            If True, compute total log-likelihood, AIC, BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If not computing metrics: optimal parameter dict.
            If computing metrics and converged: (optimal_params, aic, bic, log_likelihood_value, standard_errors).
        """
        if initial_params is None:
            initial_params = {"mu": float(np.mean(self.kernel)), "phi": 0.1, "delta": 0.5, "v1": 5.0, "v2": 5.0}

        keys = list(initial_params.keys())
        x0 = np.array(list(initial_params.values()), dtype=float)
        n = len(self.kernel) - 1

        bounds = [(-np.inf, np.inf), (-0.999, 0.999), (1e-6, 1.0), (2.000001, None), (2.000001, None)]
        res = scipy.optimize.minimize(self.log_likelihood, x0, method="SLSQP", bounds=bounds)

        self.convergence = bool(res.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, res.x))

        if compute_metrics and self.convergence:
            tot_ll = -res.fun * n
            self.log_likelihood_value = float(tot_ll)
            k = len(x0)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, k)

            H = approx_hess1(res.x, self.log_likelihood, args=())
            cov = inv(H) / n
            ses = np.sqrt(np.diag(cov))
            if np.isnan(ses).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = approx_fprime(res.x, self.log_likelihood, eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.diag(cov_alt))
            self.standard_errors = ses.astype(float)
            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)

        return self.optimal_params

    def one_step_ahead_forecast(self) -> float:
        """
        Compute the one-step-ahead conditional mean forecast.

        Returns
        -------
        float
            Forecast for y_{T+1} given data up to time T.

        Raises
        ------
        RuntimeError
            If the model has not been optimized yet.
        """
        if self.optimal_params is None:
            raise RuntimeError("Optimize model first.")
        mu = self.optimal_params["mu"]
        phi = self.optimal_params["phi"]
        return float(mu + phi * (self.kernel[-1] - mu))

    def multi_step_ahead_forecast(self, horizon: int) -> NDArray[np.float64]:
        """
        Compute multi-step-ahead conditional mean forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead (h >= 1).

        Returns
        -------
        np.ndarray
            Array of length `horizon` with forecasts.

        Raises
        ------
        RuntimeError
            If the model has not been optimized yet.
        """
        if self.optimal_params is None:
            raise RuntimeError("Optimize model first.")
        mu = self.optimal_params["mu"]
        phi = self.optimal_params["phi"]
        last = self.kernel[-1]
        forecasts = np.empty(horizon, dtype=float)
        forecasts[0] = mu + phi * (last - mu)
        for h in range(1, horizon):
            forecasts[h] = mu + phi * (forecasts[h - 1] - mu)
        return forecasts
