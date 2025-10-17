from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import scipy.optimize as opt
from scipy.special import gammaln, gamma  # kept for parity with original imports
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1


class GARCHMIDASASTModel:
    """
    GARCH-MIDAS with Asymmetric Student-t (AST) innovations.

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
        Innovation distribution ("AST").
    log_returns : np.ndarray
        Original returns array.
    X : np.ndarray
        MIDAS regressor matrix with shape (T, K-1).
    K : int
        MIDAS lag count.
    T : int
        Effective sample size after alignment.
    inflation_series : np.ndarray
        Inflation series storage.
    inflation_dates : np.ndarray
        Inflation dates storage.
    daily_dates : np.ndarray
        Daily dates storage.
    r : np.ndarray
        Returns aligned to the K-lag mapping (length T).
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters if `optimize` has been called successfully.
    convergence : bool
        Optimizer success flag.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (sum over effective observations).
    aic : Optional[float]
        Akaike Information Criterion at optimum.
    bic : Optional[float]
        Bayesian Information Criterion at optimum.
    standard_errors : Optional[np.ndarray]
        Approximate standard errors from inverse Hessian (when computed).
    """

    model_name: str = "GARCHMIDAS-AST"
    distribution: str = "AST"

    log_returns: np.ndarray
    X: np.ndarray
    K: int
    T: int
    inflation_series: np.ndarray
    inflation_dates: np.ndarray
    daily_dates: np.ndarray
    r: np.ndarray

    optimal_params: Optional[Dict[str, float]]
    convergence: bool
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    standard_errors: Optional[np.ndarray]

    def __init__(self, log_returns: np.ndarray, inflation_series: np.ndarray, inflation_dates: np.ndarray, daily_dates: np.ndarray, K: int) -> None:
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.X = None  # will be set below
        self.K = int(K)
        self.T = int(len(self.log_returns))
        self.inflation_series = np.asarray(inflation_series, dtype=float)
        self.inflation_dates = np.asarray(inflation_dates)
        self.daily_dates = np.asarray(daily_dates)

        self.optimal_params = None
        self.convergence = False
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.standard_errors = None

        idx = np.searchsorted(self.inflation_dates, self.daily_dates, side="right") - 1
        lag = self.K - 1
        idx = idx[lag:]
        self.r = self.log_returns[lag:]
        self.T = int(len(self.r))

        Xmat = np.zeros((self.T, self.K), dtype=float)
        for t in range(self.T):
            j = idx[t]
            i0 = max(j - (self.K - 1), 0)
            block = self.inflation_series[i0 : j + 1]
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
        numer = (1.0 - j / K) ** (w - 1.0)
        denom = float(np.sum(numer))
        return numer / denom

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
        beta_weights = self.betapolyn(self.K, w)
        return m**2 + theta**2 * (self.X @ beta_weights)

    def _K(self, v: float) -> float:
        """
        AST helper: t-kernel normalization K(v).

        Parameters
        ----------
        v : float
            Degrees of freedom (> 0).

        Returns
        -------
        float
            K(v) constant.
        """
        return float(gamma((v + 1.0) / 2.0) / (np.sqrt(np.pi * v) * gamma(v / 2.0)))

    def B(self, delta: float, v1: float, v2: float) -> float:
        """
        AST mixture weight B(delta, v1, v2).

        Parameters
        ----------
        delta : float
            Mixing parameter in [0, 1].
        v1, v2 : float
            Left/right degrees of freedom (> 2 for finite variance).

        Returns
        -------
        float
            Mixture normalization constant.
        """
        return float(delta * self._K(v1) + (1.0 - delta) * self._K(v2))

    def alpha_star(self, delta: float, v1: float, v2: float) -> float:
        """
        AST auxiliary α* (skew weight).

        Returns
        -------
        float
            Alpha-star parameter in (0, 1).
        """
        B = self.B(delta, v1, v2)
        return float((delta * self._K(v1)) / B)

    def m_s(self, delta: float, v1: float, v2: float) -> Tuple[float, float]:
        """
        AST location-scale moments (m, s).

        Parameters
        ----------
        delta : float
            Mixing parameter in [0, 1].
        v1, v2 : float
            Left/right d.f. (> 2).

        Returns
        -------
        (float, float)
            Tuple of (m, s).
        """
        B = self.B(delta, v1, v2)
        a_star = self.alpha_star(delta, v1, v2)
        m = 4.0 * B * (-(a_star**2) * v1 / (v1 - 1.0) + (1.0 - a_star) ** 2 * v2 / (v2 - 1.0))
        s = np.sqrt(4.0 * (delta * a_star**2 * v1 / (v1 - 2.0) + (1.0 - delta) * (1.0 - a_star) ** 2 * v2 / (v2 - 2.0)) - m**2)
        return float(m), float(s)

    def log_likelihood(self, params: Sequence[float]) -> float:
        """
        Average negative log-likelihood under AST innovations.

        Parameters
        ----------
        params : sequence of float
            (mu, alpha, beta, theta, w, m_param, delta, v1, v2).

        Returns
        -------
        float
            Average negative log-likelihood (for minimization).
        """
        mu, alpha, beta, theta, w, m_param, delta, v1, v2 = params
        tau = self.tau_series(theta, w, m_param)

        g = np.empty(self.T, dtype=float)
        g[0] = 1.0
        for t in range(1, self.T):
            g[t] = (1.0 - alpha - beta) + alpha * ((self.r[t - 1] - mu) ** 2) / tau[t - 1] + beta * g[t - 1]

        sigma2 = tau * g
        z = (self.r - mu) / np.sqrt(sigma2)

        m_ast, s_ast = self.m_s(delta, v1, v2)
        ll = np.empty(self.T, dtype=float)
        K1 = self._K(v1); K2 = self._K(v2)
        Bv = self.B(delta, v1, v2)
        const = np.log(Bv) - 0.5 * np.log(sigma2)

        for t in range(self.T):
            score = m_ast + s_ast * z[t]
            if score <= 0.0:
                term = (v1 + 1.0) / 2.0 * np.log(1.0 + (score * score) / v1)
                ll[t] = np.log(K1) + const[t] - term
            else:
                term = (v2 + 1.0) / 2.0 * np.log(1.0 + (score * score) / v2)
                ll[t] = np.log(K2) + const[t] - term

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
            {'mu':0, 'alpha':0.1, 'beta':0.85, 'theta':0.1, 'w':10,
             'm_param':0.001, 'delta':0.5, 'v1':5.0, 'v2':5.0}
        compute_metrics : bool, default False
            If True, compute total log-likelihood, AIC/BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If `compute_metrics` and convergence:
            (params, AIC, BIC, loglik, SEs); else params dict.
        """
        if initial_params is None:
            initial_params = self.optimal_params if self.optimal_params is not None else {'mu': 0.0, 'alpha': 0.1, 'beta': 0.85, 'theta': 0.1, 'w': 10.0, 'm_param': 0.001, 'delta': 0.5, 'v1': 5.0, 'v2': 5.0}

        keys = list(initial_params.keys())
        x0 = np.asarray(list(initial_params.values()), dtype=float)
        bounds = [(-np.inf, np.inf), (0.0, 1.0), (0.0, 1.0), (0.0, None), (1.0, None), (0.0, None), (0.0, 1.0), (2.001, None), (2.001, None)]
        cons = {'type': 'ineq', 'fun': lambda x: 1.0 - x[1] - x[2]}

        res = opt.minimize(self.log_likelihood, x0, method='SLSQP', bounds=bounds, constraints=cons)
        self.convergence = bool(res.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))

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
        Multi-step-ahead variance forecasts using GARCH-MIDAS recursion (AST errors).

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
            raise ValueError("Model must be optimized before forecasting.")
        mu = float(self.optimal_params['mu']); alpha = float(self.optimal_params['alpha']); beta = float(self.optimal_params['beta'])
        theta = float(self.optimal_params['theta']); w = float(self.optimal_params['w']); m_param = float(self.optimal_params['m_param'])

        r_T = float(self.r[-1])
        tau = self.tau_series(theta, w, m_param)
        g = np.empty(self.T, dtype=float)
        g[0] = 1.0
        for t in range(1, self.T):
            g[t] = (1.0 - alpha - beta) + alpha * ((self.r[t - 1] - mu) ** 2) / tau[t - 1] + beta * g[t - 1]

        beta_wts = self.betapolyn(self.K, w)
        X_last = self.X[-1]
        tau_next = m_param**2 + theta**2 * (X_last @ beta_wts)

        g1 = (1.0 - alpha - beta) + alpha * (r_T - mu) ** 2 / tau[-1] + beta * g[-1]
        sigma2_1 = float(tau_next * g1)

        forecasted_vars = [sigma2_1]
        for _ in range(1, int(horizon)):
            next_var = (1.0 - alpha - beta) * tau_next + (alpha + beta) * forecasted_vars[-1]
            forecasted_vars.append(float(next_var))
        return np.asarray(forecasted_vars, dtype=float)
