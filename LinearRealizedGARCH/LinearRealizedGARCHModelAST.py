from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.special import gamma
from statsmodels.tools.numdiff import approx_fprime, approx_hess1


class LinearRealizedGARCHModelAST:
    """
    Linear Realized GARCH(1,1) with Asymmetric Student-t (AST) innovations.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of returns.
    x : np.ndarray
        1-D array of realized measures aligned with `log_returns`.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Error distribution ('AST').
    log_returns : np.ndarray
        Return series as float array.
    x : np.ndarray
        Realized measure series as float array.
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

    model_name: str = "LRGARCH-AST"
    distribution: str = "AST"

    log_returns: np.ndarray
    x: np.ndarray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: Optional[bool]
    standard_errors: Optional[np.ndarray]

    def __init__(self, log_returns: np.ndarray, x: np.ndarray) -> None:
        """Initialize the model with input series."""
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = None
        self.standard_errors = None

    def K(self, v: float) -> float:
        """K(v) helper for AST density constant."""
        return float(gamma((v + 1.0) / 2.0) / (np.sqrt(np.pi * v) * gamma(v / 2.0)))

    def B(self, delta: float, v1: float, v2: float) -> float:
        """B(delta, v1, v2) mixture weight for AST."""
        return float(delta * self.K(v1) + (1.0 - delta) * self.K(v2))

    def alpha_star(self, delta: float, v1: float, v2: float) -> float:
        """α* (skew weight) for AST."""
        B_val = self.B(delta, v1, v2)
        return float((delta * self.K(v1)) / B_val)

    def m(self, delta: float, v1: float, v2: float) -> float:
        """AST mean-adjustment m(delta, v1, v2)."""
        a = self.alpha_star(delta, v1, v2)
        B_val = self.B(delta, v1, v2)
        return float(4.0 * B_val * (-(a**2) * v1 / (v1 - 1.0) + (1.0 - a) ** 2 * v2 / (v2 - 1.0)))

    def s(self, delta: float, v1: float, v2: float) -> float:
        """AST scale-adjustment s(delta, v1, v2)."""
        a = self.alpha_star(delta, v1, v2)
        m_val = self.m(delta, v1, v2)
        return float(np.sqrt(4.0 * (delta * a**2 * v1 / (v1 - 2.0) + (1.0 - delta) * (1.0 - a) ** 2 * v2 / (v2 - 2.0)) - m_val**2))

    def I_t(self, r_t: float, mu_t: float, h_t: float, m_val: float, s_val: float) -> int:
        """AST indicator I(z>0)."""
        return 1 if m_val + s_val * (r_t - mu_t) / np.sqrt(h_t) > 0.0 else 0

    def log_likelihood(self, params: Dict[str, float]) -> float:
        """Average negative log-likelihood under AST for returns and Gaussian for x|h,z,tau.

        Parameters
        ----------
        params : dict
            {'omega','beta','gamma','xi','phi','tau_1','tau_2','sigma_u','delta','v1','v2'}.

        Returns
        -------
        float
            Average negative log-likelihood (to minimize).
        """
        T = len(self.log_returns)
        omega = params["omega"]; beta = params["beta"]; gamma_ = params["gamma"]; xi = params["xi"]; phi = params["phi"]
        tau_1 = params["tau_1"]; tau_2 = params["tau_2"]; sigma_u = params["sigma_u"]; delta = params["delta"]; v1 = params["v1"]; v2 = params["v2"]

        mu_t = 0.0
        h = np.zeros(T, dtype=float); z = np.zeros(T, dtype=float); tau = np.zeros(T, dtype=float); u = np.zeros(T, dtype=float); I_val = np.zeros(T, dtype=float)
        h[0] = float(np.var(self.log_returns[:50]) if T >= 1 else 1.0)

        for t in range(1, T): h[t] = omega + beta * h[t - 1] + gamma_ * self.x[t - 1]
        for t in range(T):
            if h[t] <= 0.0: return 1e10
            z[t] = self.log_returns[t] / np.sqrt(h[t])
            tau[t] = tau_1 * z[t] + tau_2 * (z[t] ** 2 - 1.0)
            u[t] = self.x[t] - xi - phi * h[t] - tau[t]
            m_val = self.m(delta, v1, v2); s_val = self.s(delta, v1, v2); a_star = self.alpha_star(delta, v1, v2)
            I_val[t] = float(self.I_t(self.log_returns[t], mu_t, h[t], m_val, s_val))
            B_val = self.B(delta, v1, v2)

        m_val = self.m(delta, v1, v2); s_val = self.s(delta, v1, v2); a_star = self.alpha_star(delta, v1, v2); B_val = self.B(delta, v1, v2)
        term1 = (v1 + 1.0) / 2.0 * np.log(1.0 + (1.0 / v1) * ((m_val + s_val * (self.log_returns - mu_t) / np.sqrt(h)) / (2.0 * a_star)) ** 2)
        term2 = (v2 + 1.0) / 2.0 * np.log(1.0 + (1.0 / v2) * ((m_val + s_val * (self.log_returns - mu_t) / np.sqrt(h)) / (2.0 * (1.0 - a_star))) ** 2)
        loglik_r = np.log(s_val) + np.log(B_val) - 0.5 * np.log(h) - (1.0 - I_val) * term1 - I_val * term2

        if sigma_u <= 0.0 or not np.isfinite(sigma_u): return 1e10
        loglik_x = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma_u**2) + (u**2) / (sigma_u**2))

        return float(-np.mean(loglik_r + loglik_x))

    def compute_aic_bic(self, log_likelihood: float, num_params: int) -> Tuple[float, float]:
        """Compute AIC and BIC (preserving original scaling)."""
        n = len(self.log_returns)
        aic = 2.0 * num_params - 2.0 * log_likelihood * n
        bic = np.log(n) * num_params - 2.0 * log_likelihood * n
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """Estimate parameters by SLSQP with bounds and stability constraint.

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum or default starting values from the original code.
        compute_metrics : bool, default False
            If True, also compute AIC/BIC, total log-likelihood, and standard errors.

        Returns
        -------
        dict or tuple
            Params dict; or (params, AIC, BIC, total_loglik, SEs) when `compute_metrics=True`.
        """
        if self.optimal_params is not None:
            initial_params = self.optimal_params
        elif initial_params is None:
            initial_params = {
                "omega": 0.06, "beta": 0.55, "gamma": 0.41, "xi": -0.18, "phi": 1.04,
                "tau_1": -0.07, "tau_2": 0.07, "sigma_u": 0.38, "delta": 0.3, "v1": 4.130, "v2": 4.130,
            }

        keys = list(initial_params.keys())
        x0 = [float(v) for v in initial_params.values()]
        cons = ({'type': 'ineq', 'fun': lambda x: 1.0 - x[1] - x[2] * x[4]})
        bounds = [
            (1e-5, np.inf), (0.0, np.inf), (0.0, np.inf), (1e-5, np.inf), (0.0, np.inf),
            (-np.inf, np.inf), (-np.inf, np.inf), (1e-5, np.inf), (1e-5, 0.99999), (2.00001, np.inf), (2.00001, np.inf),
        ]
        options = {'disp': True, 'maxiter': 200, 'eps': 1e-9}

        obj = lambda arr: self.log_likelihood(dict(zip(keys, arr)))
        result = minimize(obj, x0, method='SLSQP', bounds=bounds, options=options, constraints=cons)

        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in result.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")
        if compute_metrics and self.convergence:
            self.log_likelihood_value = float(-result.fun * len(self.log_returns))
            k = len(self.optimal_params)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, k)

            f_for_hess = lambda arr: self.log_likelihood(dict(zip(keys, arr)))
            hess = approx_hess1(np.asarray(list(self.optimal_params.values()), dtype=float), f_for_hess)
            cov = inv(hess) / max(len(self.log_returns), 1)
            ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            if np.isnan(ses).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = approx_fprime(np.asarray(list(self.optimal_params.values()), dtype=float), f_for_hess, eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.maximum(np.diag(cov_alt), 0.0))
            self.standard_errors = ses
            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors
        return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """Multi-step forecasts of conditional variance h_t holding x_t at its last observed value.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecasted variances [h_{T+1}, …, h_{T+horizon}].
        """
        if self.optimal_params is None: raise ValueError("Model must be optimized before forecasting.")

        omega = self.optimal_params["omega"]; beta = self.optimal_params["beta"]; gamma_ = self.optimal_params["gamma"]
        T = len(self.log_returns)
        h = np.zeros(T, dtype=float)
        h[0] = float(np.var(self.log_returns[:50]) if T >= 1 else 1.0)
        for t in range(1, T): h[t] = omega + beta * h[t - 1] + gamma_ * self.x[t - 1]
        h_last = float(h[-1]); x_last = float(self.x[-1])

        forecasts: List[float] = []
        prev = h_last
        for _ in range(int(horizon)):
            nxt = float(omega + beta * prev + gamma_ * x_last)
            forecasts.append(nxt)
            prev = nxt
        return np.asarray(forecasts, dtype=float)
