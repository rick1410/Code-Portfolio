from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma, digamma, polygamma
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.optimize import minimize, approx_fprime

class AR1EGB2Model:
    """
    AR(1) model with EGB2 innovations, estimated by MLE.

    Model
    -----
    r_t = phi r_{t-1} + ε_t,  ε_t ~ EGB2(p, q)

    Parameters
    ----------
    kernel_array : ArrayLike
        1-D time series (treated as log returns).

    Attributes
    ----------
    model_name : str
        Short name of the model.
    distribution : str
        Innovation distribution name ("EGB2").
    log_returns : NDArray
        Stored series as a 1-D NumPy array.
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters after optimization: {"phi", "p", "q"}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at the optimum.
    aic : Optional[float]
        Akaike Information Criterion at the optimum.
    bic : Optional[float]
        Bayesian Information Criterion at the optimum.
    convergence : bool
        Whether the optimizer reported success.
    standard_errors : Optional[NDArray[np.float64]]
        Approximate standard errors for parameters (from inverse Hessian).
    """

    model_name: str = "AR1-EGB2"
    distribution: str = "EGB2"

    log_returns: NDArray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: bool
    standard_errors: Optional[NDArray[np.float64]]

    def __init__(self, kernel_array: ArrayLike) -> None:
        """Initialize the AR(1)-EGB2 model with a time series."""
        self.log_returns = np.asarray(kernel_array)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def _egb2_constants(self, p: float, q: float) -> Tuple[float, float, float]:
        """
        Compute EGB2 helper constants.

        Parameters
        ----------
        p : float
            Shape parameter (> 0).
        q : float
            Shape parameter (> 0).

        Returns
        -------
        tuple[float, float, float]
            (Delta, Omega, norm_const) used in the log-likelihood.
        """
        Delta = float(digamma(p) - digamma(q))
        Omega = float(polygamma(1, p) + polygamma(1, q))
        norm_const = float(gamma(p) * gamma(q) / gamma(p + q))
        return Delta, Omega, norm_const

    def log_likelihood(self, params: ArrayLike) -> float:
        """
        Negative average log-likelihood for AR(1) with EGB2 innovations.

        Parameters
        ----------
        params : array-like of shape (3,)
            Parameter vector [phi, p, q].

        Returns
        -------
        float
            Negative average log-likelihood (lower is better).
        """
        phi, p, q = params
        r = self.log_returns
        eps = r[1:] - phi * r[:-1]
        N = eps.size

        Delta, Omega, norm_const = self._egb2_constants(p, q)
        root_Omega = np.sqrt(Omega)

        ll = np.zeros(N)
        for i, e in enumerate(eps):
            z = root_Omega * e + Delta
            ll[i] = 0.5 * np.log(Omega) + p * z - np.log(norm_const) - (p + q) * np.log1p(np.exp(z))
        return -np.mean(ll)

    def compute_aic_bic(self, total_loglik: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC from total log-likelihood.

        Parameters
        ----------
        total_loglik : float
            Sum log-likelihood across effective observations.
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        tuple[float, float]
            (AIC, BIC).
        """
        N = len(self.log_returns) - 1
        aic = 2 * num_params - 2 * total_loglik
        bic = np.log(N) * num_params - 2 * total_loglik
        return aic, bic

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, NDArray[np.float64]]:
        """
        Fit parameters via SLSQP with bounds; optionally compute metrics and SEs.

        Parameters
        ----------
        initial_params : dict or None
            Initial guess as {"phi": φ0, "p": p0, "q": q0}. If None, uses previous
            optimum if available, otherwise sensible defaults.
        compute_metrics : bool, default False
            If True, also computes total log-likelihood, AIC, BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If `compute_metrics` is False: dict with optimal parameters.
            If `compute_metrics` is True and converged:
                (optimal_params, aic, bic, log_likelihood_value, standard_errors).
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {"phi": 0.1, "p": 3.5, "q": 3.5}

        keys = list(initial_params.keys())
        x0 = np.array(list(initial_params.values()), dtype=float)
        N = len(self.log_returns) - 1

        bounds = [(-0.999, 0.999), (2.000001, None), (2.000001, None)]
        result = minimize(self.log_likelihood, x0, method="SLSQP", bounds=bounds)

        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, result.x))

        if compute_metrics and self.convergence:
            total_ll = -result.fun * N
            self.log_likelihood_value = float(total_ll)
            k = len(x0)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, k)

            try:
                H = approx_hess1(result.x, self.log_likelihood, args=())
                cov = inv(H) / N
                ses = np.sqrt(np.diag(cov))
                if np.isnan(ses).any():
                    raise FloatingPointError("Non-finite SEs from Hessian.")
            except Exception:
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = approx_fprime(result.x, self.log_likelihood, eps)
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
            Forecast for r_{T+1} given data up to time T.

        Raises
        ------
        ValueError
            If the model has not been optimized yet.
        """
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")
        phi = self.optimal_params.get("phi")
        return float(phi * self.log_returns[-1])

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
        ValueError
            If the model has not been optimized yet.
        """
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")
        phi = self.optimal_params.get("phi")
        last = self.log_returns[-1]
        forecasts = np.empty(horizon, dtype=float)
        forecasts[0] = phi * last
        for h in range(1, horizon):
            forecasts[h] = phi * forecasts[h - 1]
        return forecasts
