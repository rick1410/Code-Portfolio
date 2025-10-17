from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv
import scipy
from statsmodels.tools.numdiff import approx_hess1


class GARCHMIDASModel:
    """
    GARCH-MIDAS with exogenous MIDAS component from an inflation series.

    Parameters
    ----------
    log_returns : np.ndarray
        Daily log returns (1-D).
    inflation_series : np.ndarray
        Low-frequency inflation series aligned with `inflation_dates`.
    inflation_dates : np.ndarray
        Dates for `inflation_series`, sortable (e.g., datetime64).
    daily_dates : np.ndarray
        Daily dates for `log_returns`, sortable (e.g., datetime64).
    K : int
        Number of MIDAS lags (uses K−1 lags in X plus contemporaneous mapping).

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Innovation distribution ("Normal").
    log_returns : np.ndarray
        Original returns array.
    K : int
        MIDAS lag count.
    inflation_series : np.ndarray
        Inflation series storage.
    inflation_dates : np.ndarray
        Inflation dates storage.
    r : np.ndarray
        Returns aligned to the K-lag mapping (length T).
    T : int
        Effective sample size after alignment.
    X : np.ndarray
        MIDAS regressor matrix with shape (T, K-1).
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters if `optimize` has been called successfully.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum.
    aic : Optional[float]
        Akaike Information Criterion at optimum.
    bic : Optional[float]
        Bayesian Information Criterion at optimum.
    convergence : bool
        Optimizer success flag.
    standard_errors : Optional[np.ndarray]
        Approximate standard errors from inverse Hessian (if computed).
    """

    model_name: str = "GARCH-MIDAS"
    distribution: str = "Normal"

    log_returns: np.ndarray
    K: int
    inflation_series: np.ndarray
    inflation_dates: np.ndarray
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
        self.K = int(K)
        self.inflation_series = np.asarray(inflation_series, dtype=float)
        self.inflation_dates = np.asarray(inflation_dates)
        idx = np.searchsorted(self.inflation_dates, daily_dates, side="right") - 1
        lag = self.K - 1
        idx = idx[lag:]
        self.r = self.log_returns[lag:]
        self.T = int(len(self.r))
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

        Xmat = np.zeros((self.T, self.K), dtype=float)
        for t in range(self.T):
            j = idx[t]
            block = self.inflation_series[max(j - (self.K - 1), 0): j + 1]
            if block.size < self.K:
                pad = np.full(self.K - block.size, self.inflation_series[0])
                block = np.concatenate([pad, block])
            Xmat[t] = block
        self.X = Xmat[:, :-1]

    @staticmethod
    def betapolyn(K: int, w: float) -> np.ndarray:
        """
        Normalized Beta-polynomial weights for MIDAS.

        Parameters
        ----------
        K : int
            Total number of lags (produces K-1 weights).
        w : float
            Shape parameter (> 0) controlling decay.

        Returns
        -------
        np.ndarray
            Weights summing to 1 with shape (K-1,).
        """
        j = np.arange(1, K, dtype=float)
        num = (1.0 - j / K) ** (w - 1.0)
        return num / num.sum()

    def tau_series(self, theta: float, w: float, m: float) -> np.ndarray:
        """
        MIDAS long-run component τ_t = m^2 + θ^2 * (X_t ⋅ weights).

        Parameters
        ----------
        theta : float
            MIDAS scaling parameter (≥ 0).
        w : float
            Beta-polynomial shape (> 0).
        m : float
            Long-run mean level (≥ 0).

        Returns
        -------
        np.ndarray
            τ_t series of length T.
        """
        weights = self.betapolyn(self.K, w)
        return m**2 + theta**2 * (self.X @ weights)

    def log_likelihood(self, params: Sequence[float]) -> float:
        """
        Average negative log-likelihood under conditional Normality.

        Parameters
        ----------
        params : sequence of float
            (mu, alpha, beta, theta, w, m).

        Returns
        -------
        float
            Average negative log-likelihood (for minimization).
        """
        mu, alpha, beta, theta, w, m = params
        tau = self.tau_series(theta, w, m)
        g = np.empty(self.T, dtype=float)
        g[0] = 1.0
        for t in range(1, self.T):
            g[t] = (1.0 - alpha - beta + alpha * ((self.r[t - 1] - mu) ** 2) / tau[t - 1] + beta * g[t - 1])
        sigma2 = tau * g
        ll = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + (self.r - mu) ** 2 / sigma2)
        return float(-np.mean(ll))

    def compute_aic_bic(self, total_ll: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC from total log-likelihood.

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
        aic = 2.0 * num_params - 2.0 * total_ll
        bic = np.log(self.T) * num_params - 2.0 * total_ll
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, np.ndarray]:
        """
        Fit parameters via SLSQP with stationarity/positivity constraints.

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum if available else:
            {'mu': 0, 'alpha': 0.1, 'beta': 0.85, 'theta': 0.1, 'w': 10, 'm': 0.001}
        compute_metrics : bool, default False
            If True, compute total log-likelihood, AIC/BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If `compute_metrics` and convergence: (params, AIC, BIC, loglik, SEs); else params dict.
        """
        if initial_params is None:
            initial_params = self.optimal_params if self.optimal_params is not None else {'mu': 0.0, 'alpha': 0.1, 'beta': 0.85, 'theta': 0.1, 'w': 10.0, 'm': 0.001}

        keys = list(initial_params.keys()); x0 = list(initial_params.values())
        bounds = [(-np.inf, np.inf), (0.0, 1.0), (0.0, 1.0), (0.0, None), (1.0, None), (0.0, None)]
        cons = {'type': 'ineq', 'fun': lambda x: 1.0 - x[1] - x[2]}

        res = opt.minimize(self.log_likelihood, x0, method='SLSQP', bounds=bounds, constraints=cons)
        self.convergence = bool(res.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))
        else:
            print(f"Warning: Optimization failed for {self.model_name}")

        if compute_metrics and self.convergence:
            total_ll = float(-res.fun * self.T)
            self.log_likelihood_value = total_ll
            k = len(x0)
            self.aic, self.bic = self.compute_aic_bic(total_ll, k)
            H = approx_hess1(np.asarray(res.x, dtype=float), self.log_likelihood, args=())
            cov = inv(H) / self.T
            ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            if not np.all(np.isfinite(ses)):
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = opt.approx_fprime(np.asarray(res.x, dtype=float), self.log_likelihood, eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.maximum(np.diag(cov_alt), 0.0))
            self.standard_errors = ses.astype(float)
            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)
        return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Multi-step-ahead variance forecasts using GARCH-MIDAS recursion.

        Parameters
        ----------
        horizon : int
            Number of steps ahead (≥ 1).

        Returns
        -------
        np.ndarray
            Array of length `horizon` with variance forecasts.
        """
        if self.optimal_params is None:
            raise RuntimeError("Optimize model before forecasting.")
        mu, alpha, beta, theta, w, m = [self.optimal_params[k] for k in ['mu', 'alpha', 'beta', 'theta', 'w', 'm']]
        tau = self.tau_series(theta, w, m)
        g = np.empty(self.T, dtype=float)
        g[0] = 1.0
        for t in range(1, self.T):
            g[t] = (1.0 - alpha - beta + alpha * ((self.r[t - 1] - mu) ** 2) / tau[t - 1] + beta * g[t - 1])
        rT = self.r[-1]

        beta_wts = self.betapolyn(self.K, w)
        tau_next = m**2 + theta**2 * (self.X[-1] @ beta_wts)
        g1 = (1.0 - alpha - beta + alpha * (rT - mu) ** 2 / tau[-1] + beta * g[-1])
        forecasts = [float(tau_next * g1)]
        for _ in range(1, int(horizon)):
            next_var = (1.0 - alpha - beta) * tau_next + (alpha + beta) * forecasts[-1]
            forecasts.append(float(next_var))
        return np.asarray(forecasts, dtype=float)
