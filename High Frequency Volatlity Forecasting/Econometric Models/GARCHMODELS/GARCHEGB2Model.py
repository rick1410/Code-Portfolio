from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import scipy
from scipy.special import digamma, polygamma
from scipy.special import gamma as sp_gamma
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1


class GARCHEGB2Model:
    """
    GARCH(1,1) with EGB2 innovations.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of returns.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Innovation distribution ("EGB2").
    log_returns : np.ndarray
        Stored return series.
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters if `optimize` succeeds with keys:
        {'omega','alpha','beta','p','q'}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (sum over observations).
    aic : Optional[float]
        Akaike Information Criterion at optimum.
    bic : Optional[float]
        Bayesian Information Criterion at optimum.
    convergence : bool
        Optimizer success flag.
    standard_errors : Optional[np.ndarray]
        Approximate standard errors from inverse Hessian (when computed).
    """

    model_name: str = "GARCH-EGB2"
    distribution: str = "EGB2"

    log_returns: np.ndarray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: bool
    standard_errors: Optional[np.ndarray]

    def __init__(self, log_returns: np.ndarray) -> None:
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def log_likelihood(self, params: Sequence[float]) -> float:
        """
        Average negative log-likelihood under EGB2 errors for GARCH(1,1).

        Parameters
        ----------
        params : sequence of float
            (omega, alpha, beta, p, q).

        Returns
        -------
        float
            Average negative log-likelihood (for minimization).
        """
        omega, alpha, beta, p, q = params
        r = self.log_returns
        T = len(r)

        sigma2 = np.empty(T, dtype=float)
        sigma2[0] = float(np.var(r[:50])) if T > 50 else float(np.var(r))

        Delta = digamma(p) - digamma(q)
        Omega = polygamma(1, p) + polygamma(1, q)
        norm_const = sp_gamma(p) * sp_gamma(q) / sp_gamma(p + q)

        for t in range(1, T):
            sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
            if sigma2[t] <= 0.0:
                return 1e6

        ll = np.empty(T, dtype=float)
        root_O = float(np.sqrt(Omega))
        for t in range(T):
            z = root_O * r[t] / np.sqrt(sigma2[t]) + Delta
            ll[t] = (0.5 * np.log(Omega) + p * z - np.log(norm_const) - (p + q) * np.log1p(np.exp(z)) - 0.5 * np.log(sigma2[t]))
        return float(-np.mean(ll))

    def compute_aic_bic(self, total_ll: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC/BIC from total log-likelihood.

        Parameters
        ----------
        total_ll : float
            Sum of log-likelihood contributions.
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        (float, float)
            (AIC, BIC).
        """
        T = len(self.log_returns)
        aic = 2.0 * num_params - 2.0 * total_ll
        bic = np.log(T) * num_params - 2.0 * total_ll
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, np.ndarray]:
        """
        Fit parameters via SLSQP with bounds and stationarity constraint.

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum if available else defaults:
            {'omega': var(r[:50])*0.2, 'alpha': 0.1, 'beta': 0.7, 'p': 3.5, 'q': 3.5}.
        compute_metrics : bool, default False
            If True, compute total log-likelihood, AIC/BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If `compute_metrics` and convergence:
            (params, AIC, BIC, loglik, SEs); else params dict.
        """
        if initial_params is None:
            initial_params = self.optimal_params or {'omega': float(np.var(self.log_returns[:50]) * 0.2), 'alpha': 0.1, 'beta': 0.7, 'p': 3.5, 'q': 3.5}
        keys = list(initial_params.keys()); x0 = list(initial_params.values())

        bounds = [(1e-6, None), (0.0, None), (0.0, None), (2.000001, None), (2.000001, None)]
        cons = {'type': 'ineq', 'fun': lambda x: 1.0 - x[1] - x[2]}

        res = scipy.optimize.minimize(self.log_likelihood, x0, method='SLSQP', bounds=bounds, constraints=cons)
        self.convergence = bool(res.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))
        else:
            print(f"Warning: Optimization failed for {self.model_name}.")
        if compute_metrics and self.convergence:
            T = len(self.log_returns)
            total_ll = float(-res.fun * T)
            self.log_likelihood_value = total_ll
            k = len(x0)
            self.aic, self.bic = self.compute_aic_bic(total_ll, k)

            H = approx_hess1(np.asarray(res.x, dtype=float), self.log_likelihood, args=())
            cov = inv(H) / T
            ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            if not np.all(np.isfinite(ses)):
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = scipy.optimize.approx_fprime(np.asarray(res.x, dtype=float), self.log_likelihood, eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.maximum(np.diag(cov_alt), 0.0))
            self.standard_errors = ses.astype(float)
            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)
        return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int = 5) -> np.ndarray:
        """
        Multi-step variance forecast by GARCH recursion.

        Parameters
        ----------
        horizon : int, default 5
            Number of steps ahead.

        Returns
        -------
        np.ndarray
            Forecasted conditional variances of length `horizon`.
        """
        params = self.optimal_params
        r = self.log_returns
        T = len(r)

        sigma2 = np.empty(T, dtype=float)
        sigma2[0] = float(np.var(r[:50])) if T > 50 else float(np.var(r))
        for t in range(1, T):
            sigma2[t] = params['omega'] + params['alpha'] * (r[t - 1] ** 2) + params['beta'] * sigma2[t - 1]

        last = float(sigma2[-1])
        f1 = params['omega'] + params['alpha'] * (r[-1] ** 2) + params['beta'] * last
        forecasts = [f1]
        for _ in range(1, int(horizon)):
            f1 = params['omega'] + (params['alpha'] + params['beta']) * f1
            forecasts.append(f1)
        return np.array(forecasts, dtype=float)
