from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize  # kept for parity with original imports
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.special import gamma as sp_gamma, digamma  # digamma kept for parity
import scipy


class GARCHastModel:
    """
    GARCH with Asymmetric Student-t (AST) innovations.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of returns.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Innovation distribution ("AST").
    log_returns : np.ndarray
        Stored return series.
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters if `optimize` succeeds with keys:
        {'omega','alpha','beta','delta','v1','v2'}.
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

    model_name: str = "GARCH-AST"
    distribution: str = "AST"

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

    def _K(self, v: float) -> float:
        """K(v) constant for Student-t density piece."""
        return float(sp_gamma((v + 1.0) / 2.0) / (np.sqrt(np.pi * v) * sp_gamma(v / 2.0)))

    def B(self, delta: float, v1: float, v2: float) -> float:
        """Mixture normalizer B(δ; v1, v2)."""
        return float(delta * self._K(v1) + (1.0 - delta) * self._K(v2))

    def alpha_star(self, delta: float, v1: float, v2: float) -> float:
        """α* mixing weight inside AST pieces."""
        Bv = self.B(delta, v1, v2)
        return float((delta * self._K(v1)) / Bv)

    def m_s(self, delta: float, v1: float, v2: float) -> Tuple[float, float]:
        """Location (m) and scale (s) for AST transformation."""
        Bv = self.B(delta, v1, v2)
        a = self.alpha_star(delta, v1, v2)
        m = 4.0 * Bv * (-(a * a) * v1 / (v1 - 1.0) + ((1.0 - a) ** 2) * v2 / (v2 - 1.0))
        s = float(np.sqrt(4.0 * (delta * (a * a) * v1 / (v1 - 2.0) + (1.0 - delta) * ((1.0 - a) ** 2) * v2 / (v2 - 2.0)) - m * m))
        return float(m), s

    def I_t(self, r_t: float, mu: float, h_t: float, m: float, s: float) -> int:
        """Piecewise AST indicator 1{ m + s * (r_t - mu)/sqrt(h_t) > 0 }."""
        return 1 if (m + s * (r_t - mu) / np.sqrt(h_t)) > 0.0 else 0

    def log_likelihood(self, params: Sequence[float]) -> float:
        """
        Average negative log-likelihood under AST innovations.

        Parameters
        ----------
        params : sequence of float
            (omega, alpha, beta, delta, v1, v2).

        Returns
        -------
        float
            Average negative log-likelihood (for minimization).
        """
        omega, alpha, beta, delta, v1, v2 = params
        r = self.log_returns
        T = len(r)
        mu = 0.0

        sigma2 = np.empty(T, dtype=float)
        sigma2[0] = float(np.var(r[:50])) if T > 50 else float(np.var(r))
        for t in range(1, T):
            sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
            if sigma2[t] <= 0.0:
                return 1e6

        m_ast, s_ast = self.m_s(delta, v1, v2)
        Bv = self.B(delta, v1, v2)
        ll = np.empty(T, dtype=float)
        for t in range(T):
            h = sigma2[t]
            z = (r[t] - mu) / np.sqrt(h)
            score = m_ast + s_ast * z
            const = np.log(max(s_ast, np.finfo(float).eps)) + np.log(Bv) - 0.5 * np.log(h)
            if score <= 0.0:
                ll[t] = const - ((v1 + 1.0) / 2.0) * np.log1p((score * score) / v1)
            else:
                ll[t] = const - ((v2 + 1.0) / 2.0) * np.log1p((score * score) / v2)
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
        Fit parameters via SLSQP with positivity/stationarity constraints.

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum if available else defaults:
            {'omega': 0.181, 'alpha': 0.037, 'beta': 0.916, 'delta': 0.5, 'v1': 3.0, 'v2': 3.0}.
        compute_metrics : bool, default False
            If True, compute total log-likelihood, AIC/BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If `compute_metrics` and convergence:
            (params, AIC, BIC, loglik, SEs); else params dict.
        """
        if initial_params is None:
            initial_params = self.optimal_params or {'omega': 0.181, 'alpha': 0.037, 'beta': 0.916, 'delta': 0.5, 'v1': 3.0, 'v2': 3.0}
        keys = list(initial_params.keys()); x0 = list(initial_params.values())

        bounds = [(1e-6, None), (0.0, None), (0.0, None), (0.0, 1.0), (2.001, None), (2.001, None)]
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
        Multi-step variance forecast using GARCH recursion.

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
        mu = 0.0
        r = self.log_returns
        T = len(r)

        sigma2 = np.empty(T, dtype=float)
        sigma2[0] = float(np.var(r[:50])) if T > 50 else float(np.var(r))
        for t in range(1, T):
            sigma2[t] = params['omega'] + params['alpha'] * (r[t - 1] ** 2) + params['beta'] * sigma2[t - 1]
        last = float(sigma2[-1])

        forecasts = [params['omega'] + params['alpha'] * (r[-1] ** 2) + params['beta'] * last]
        for _ in range(1, int(horizon)):
            forecasts.append(params['omega'] + (params['alpha'] + params['beta']) * forecasts[-1])
        return np.array(forecasts, dtype=float)
