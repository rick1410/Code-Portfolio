from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import scipy.optimize as opt
from scipy.special import gammaln
from statsmodels.tools.numdiff import approx_hess1
from numpy.linalg import inv


class GASMIDAStModel:
    """
    GAS-MIDAS with Student-t innovations.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of returns.
    inflation_series : np.ndarray
        Low-frequency MIDAS regressor series.
    inflation_dates : np.ndarray
        Datetime-like array for `inflation_series`.
    daily_dates : np.ndarray
        Datetime-like array for `log_returns`.
    K : int
        Number of MIDAS lags (weights length); effective regressors are K-1.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Innovation distribution ("Student t").
    log_returns : np.ndarray
        Original return series.
    inflation_series : np.ndarray
        Stored low-frequency regressor series.
    inflation_dates : np.ndarray
        Dates for the regressor series.
    daily_dates : np.ndarray
        Dates for the daily return series.
    K : int
        Number of MIDAS lags.
    r : np.ndarray
        Returns aligned with K-lagged blocks.
    T : int
        Effective sample size after alignment.
    X : np.ndarray
        MIDAS regressor matrix of shape (T, K-1).
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters with keys {"mu","alpha","beta","theta","w","m","nu1"}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (sum across observations).
    aic : Optional[float]
        Akaike Information Criterion.
    bic : Optional[float]
        Bayesian Information Criterion.
    convergence : bool
        Optimizer success flag.
    standard_errors : Optional[np.ndarray]
        Approximate standard errors from inverse Hessian (when computed).
    """

    model_name: str = "GASMIDAS-t"
    distribution: str = "Student t"

    log_returns: np.ndarray
    inflation_series: np.ndarray
    inflation_dates: np.ndarray
    daily_dates: np.ndarray
    K: int
    r: np.ndarray
    T: int
    X: np.ndarray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: bool
    standard_errors: Optional[np.ndarray]

    def __init__(self, log_returns: np.ndarray, inflation_series: np.ndarray, inflation_dates: np.ndarray, daily_dates: np.ndarray, K: int) -> None:
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.X = None  # filled below
        self.K = int(K)
        self.T = len(self.log_returns)
        self.inflation_series = np.asarray(inflation_series, dtype=float)
        self.inflation_dates = np.asarray(inflation_dates)
        self.daily_dates = np.asarray(daily_dates)
        self.convergence = False

        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.standard_errors = None

        idx = np.searchsorted(self.inflation_dates, self.daily_dates, side="right") - 1
        lag = self.K - 1
        idx = idx[lag:]
        self.r = self.log_returns[lag:]
        self.T = len(self.r)

        Xmat = np.zeros((self.T, self.K), dtype=float)
        for t in range(self.T):
            j = idx[t]
            i0 = max(j - (self.K - 1), 0)
            block = self.inflation_series[i0 : j + 1]
            if block.size < self.K:
                pad = np.full(self.K - block.size, self.inflation_series[0], dtype=float)
                block = np.concatenate([pad, block])
            Xmat[t] = block
        self.X = Xmat[:, :-1]

    @staticmethod
    def betapolyn(K: int, w: float) -> np.ndarray:
        """
        Unit-sum beta-polynomial weights of length K-1.

        Parameters
        ----------
        K : int
            Total block size (effective regressors K-1).
        w : float
            Shape parameter (> 0).

        Returns
        -------
        np.ndarray
            Weights summing to one.
        """
        j = np.arange(1, K, dtype=float)
        numer = (1.0 - j / float(K)) ** (w - 1.0)
        denom = float(np.sum(numer))
        return numer / denom

    def tau_series(self, theta: float, w: float, m: float) -> np.ndarray:
        """
        Long-run MIDAS component τ_t = m^2 + θ^2 * (X_t · weights).

        Parameters
        ----------
        theta : float
            MIDAS scale parameter (>= 0).
        w : float
            Weight-shape parameter (> 0).
        m : float
            Baseline level (>= 0).

        Returns
        -------
        np.ndarray
            τ_t series of length T.
        """
        beta_weights = self.betapolyn(self.K, w)
        return m**2 + theta**2 * (self.X @ beta_weights)

    def log_likelihood(self, params: Sequence[float]) -> float:
        """
        Average negative log-likelihood under Student's t innovations.

        Parameters
        ----------
        params : sequence of float
            (mu, alpha, beta, theta, w, m, nu).

        Returns
        -------
        float
            Average negative log-likelihood (for minimization).
        """
        mu, alpha, beta, theta, w, m, nu = params
        tau = self.tau_series(theta, w, m)

        g = np.ones(self.T, dtype=float)
        for t in range(self.T - 1):
            sigma2_t = tau[t] * g[t]
            eps = (self.r[t] - mu) / np.sqrt(max(sigma2_t, 1e-12))
            score = ( (nu + 1.0) * eps**2 / (nu + eps**2) ) - 1.0
            g[t + 1] = 1.0 + alpha * (1.0 + 3.0 / nu) * score * g[t] + beta * (g[t] - 1.0)

        sigma2 = tau * g
        const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
        ll = const - 0.5 * np.log(sigma2) - (nu + 1.0) / 2.0 * np.log(1.0 + (self.r - mu) ** 2 / (nu * sigma2))
        return float(-np.mean(ll))

    def compute_aic_bic(self, total_loglik: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC from total log-likelihood.

        Parameters
        ----------
        total_loglik : float
            Sum of log-likelihood terms.
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        (float, float)
            (AIC, BIC).
        """
        aic = 2.0 * num_params - 2.0 * total_loglik
        bic = np.log(self.T) * num_params - 2.0 * total_loglik
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, np.ndarray]:
        """
        Fit parameters via SLSQP with constraints: alpha,beta in [0,1], alpha+beta≤1, theta,m≥0, w≥1, nu>2.

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum if available else defaults:
            {'mu': mean(returns), 'alpha': 0.1, 'beta': 0.85, 'theta': 0.1, 'w': 10, 'm': 0.001, 'nu1': 5}.
        compute_metrics : bool, default False
            If True, compute total log-likelihood, AIC/BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If `compute_metrics` and convergence:
            (params, AIC, BIC, loglik, SEs); else params dict.
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {"mu": float(np.mean(self.log_returns)), "alpha": 0.1, "beta": 0.85, "theta": 0.1, "w": 10.0, "m": 0.001, "nu1": 5.0}

        keys = list(initial_params.keys()); x0 = list(initial_params.values())
        bounds = [(-np.inf, np.inf), (0.0, 1.0), (0.0, 1.0), (0.0, None), (1.0, None), (0.0, None), (2.001, None)]
        cons = {'type': 'ineq', 'fun': lambda x: 1.0 - x[1] - x[2]}

        res = opt.minimize(self.log_likelihood, x0, method='SLSQP', bounds=bounds, constraints=cons)
        self.convergence = bool(res.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")
        if compute_metrics and self.convergence:
            self.log_likelihood_value = float(-res.fun * len(self.log_returns))
            num_p = len(x0)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, num_p)

            H = approx_hess1(list(self.optimal_params.values()), self.log_likelihood, args=())
            invH = inv(H) / len(self.log_returns)
            self.standard_errors = np.sqrt(np.maximum(np.diag(invH), 0.0))
            if np.isnan(self.standard_errors).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = opt.approx_fprime(list(self.optimal_params.values()), self.log_likelihood, eps)
                invH_alt = np.outer(grad, grad)
                self.standard_errors = np.sqrt(np.maximum(np.diag(invH_alt), 0.0))

            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)
        return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Multi-step variance forecasts for GAS-MIDAS (Student t).

        Parameters
        ----------
        horizon : int
            Number of steps ahead.

        Returns
        -------
        np.ndarray
            Forecasted conditional variances of length `horizon`.
        """
        if self.optimal_params is None: raise RuntimeError("Model must be optimized before forecasting.")
        mu, alpha, beta, theta, w, m, nu = list(self.optimal_params.values())

        tau = self.tau_series(theta, w, m)
        g = 1.0
        for t in range(self.T):
            sigma2_t = tau[t] * g
            eps = (self.r[t] - mu) / np.sqrt(max(sigma2_t, 1e-12))
            score = ( (nu + 1.0) * eps**2 / (nu + eps**2) ) - 1.0
            g = 1.0 + alpha * (1.0 + 3.0 / nu) * score * g + beta * (g - 1.0)

        beta_w = self.betapolyn(self.K, w)
        Xlast = self.X[-1]
        tau_next = m**2 + theta**2 * float(Xlast @ beta_w)

        forecasts = [tau_next * g]
        g_prev = g
        for _ in range(1, int(horizon)):
            g_prev = 1.0 + beta * (g_prev - 1.0)
            forecasts.append(tau_next * g_prev)
        return np.array(forecasts, dtype=float)
