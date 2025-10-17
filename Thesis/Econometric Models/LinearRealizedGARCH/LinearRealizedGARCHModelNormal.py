from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize, approx_fprime
from statsmodels.tools.numdiff import approx_hess1


class LinearRealizedGARCHModelNormal:
    """
    Linear Realized GARCH(1,1) with Gaussian innovations for returns and
    a Gaussian measurement equation for the realized measure.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of log-returns.
    x : np.ndarray
        1-D array of realized measures (e.g., volatility proxy) aligned with `log_returns`.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Error distribution name.
    log_returns : np.ndarray
        Return series (float).
    x : np.ndarray
        Realized measure series (float).
    optimal_params : Optional[Dict[str, float]]
        Last fitted parameter set.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (mean ll * T with original sign convention).
    convergence : bool
        Optimizer success flag.
    aic : Optional[float]
        Akaike Information Criterion.
    bic : Optional[float]
        Bayesian Information Criterion.
    standard_errors : Optional[np.ndarray]
        Standard errors from inverse Hessian; may be NaN if inversion fails.
    """

    model_name: str = "LRGARCH"
    distribution: str = "Normal"

    log_returns: np.ndarray
    x: np.ndarray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    convergence: bool
    aic: Optional[float]
    bic: Optional[float]
    standard_errors: Optional[np.ndarray]

    def __init__(self, log_returns: np.ndarray, x: np.ndarray) -> None:
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.convergence = False
        self.aic = None
        self.bic = None
        self.standard_errors = None

    def log_likelihood(self, params: Dict[str, float]) -> float:
        """
        Average negative log-likelihood for the Linear Realized GARCH(1,1) with Normal errors.

        Parameters
        ----------
        params : dict
            Dictionary with parameters:
            {'omega','beta','gamma','xi','phi','tau_1','tau_2','sigma_u'}.

        Returns
        -------
        float
            Average negative log-likelihood (to minimize).
        """
        T = len(self.log_returns)
        omega = float(params["omega"])
        beta = float(params["beta"])
        gamma_ = float(params["gamma"])
        xi = float(params["xi"])
        phi = float(params["phi"])
        tau_1 = float(params["tau_1"])
        tau_2 = float(params["tau_2"])
        sigma_u = float(params["sigma_u"])

        h = np.zeros(T, dtype=float)
        z = np.zeros(T, dtype=float)
        tau = np.zeros(T, dtype=float)
        u = np.zeros(T, dtype=float)

        h[0] = float(np.var(self.log_returns[:50]) if T >= 1 else 1.0)
        for t in range(1, T): h[t] = omega + beta * h[t - 1] + gamma_ * self.x[t - 1]

        for t in range(T):
            if h[t] <= 0.0: return 1e10
            z[t] = self.log_returns[t] / np.sqrt(h[t])
            tau[t] = tau_1 * z[t] + tau_2 * (z[t] ** 2 - 1.0)
            u[t] = self.x[t] - xi - phi * h[t] - tau[t]

        if sigma_u <= 0.0 or not np.isfinite(sigma_u): return 1e10
        loglik_r = -0.5 * (np.log(2.0 * np.pi) + np.log(h) + (self.log_returns ** 2) / h)
        loglik_x = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma_u ** 2) + (u ** 2) / (sigma_u ** 2))
        return float(-np.mean(loglik_r + loglik_x))

    def calculate_aic_bic(self, log_likelihood: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC using the total log-likelihood.

        Parameters
        ----------
        log_likelihood : float
            Total log-likelihood at optimum (not average).
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        (float, float)
            AIC, BIC.
        """
        n = len(self.log_returns)
        aic = 2.0 * num_params - 2.0 * log_likelihood
        bic = np.log(n) * num_params - 2.0 * log_likelihood
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, np.ndarray]:
        """
        Estimate parameters by constrained MLE (SLSQP), preserving original behavior.

        Parameters
        ----------
        initial_params : dict or None, default None
            Starting values; if None, uses previous optimum or defaults.
        compute_metrics : bool, default False
            If True, also compute AIC, BIC, total LL, and SEs.

        Returns
        -------
        dict or tuple
            Params dict; or (params, AIC, BIC, total_LL, SEs) if `compute_metrics=True`.
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {
                    "omega": float(np.exp(0.06)),
                    "beta": 0.55,
                    "gamma": 0.41,
                    "xi": float(np.exp(-0.18)),
                    "phi": 1.04,
                    "tau_1": -0.07,
                    "tau_2": 0.07,
                    "sigma_u": 0.38,
                }

        keys = list(initial_params.keys())
        x0 = [float(v) for v in initial_params.values()]

        def constraint(arr: List[float]) -> float:
            d = dict(zip(keys, arr))
            return float(1.0 - (d["beta"] + d["phi"] * d["gamma"]))

        nonlinear_constraint = {"type": "ineq", "fun": constraint}
        bounds = [
            (1e-4, np.inf),   # omega
            (0.0, np.inf),    # beta
            (0.0, np.inf),    # gamma
            (1e-4, np.inf),   # xi
            (0.0, np.inf),    # phi
            (-np.inf, np.inf),# tau_1
            (-np.inf, np.inf),# tau_2
            (1e-5, np.inf),   # sigma_u
        ]

        obj = lambda arr: self.log_likelihood(dict(zip(keys, arr)))
        result = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=[nonlinear_constraint], options={"eps": 1e-9})

        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = {
                "omega": float(result.x[keys.index("omega")]),
                "beta": float(result.x[keys.index("beta")]),
                "gamma": float(result.x[keys.index("gamma")]),
                "xi": float(result.x[keys.index("xi")]),
                "phi": float(result.x[keys.index("phi")]),
                "tau_1": float(result.x[keys.index("tau_1")]),
                "tau_2": float(result.x[keys.index("tau_2")]),
                "sigma_u": float(result.x[keys.index("sigma_u")]),
            }
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")
            self.optimal_params = initial_params.copy()
            return self.optimal_params

        if compute_metrics and self.convergence:
            k = len(x0)
            self.log_likelihood_value = float(-result.fun * len(self.log_returns))
            self.aic, self.bic = self.calculate_aic_bic(self.log_likelihood_value, k)

            # SEs via numerical Hessian (preserving original approach)
            hess = approx_hess1(list(self.optimal_params.values()), lambda arr: self.log_likelihood(dict(zip(keys, arr))), args=())
            cov = inv(hess) / max(len(self.log_returns), 1)
            ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            if np.isnan(ses).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = approx_fprime(list(self.optimal_params.values()), lambda arr: self.log_likelihood(dict(zip(keys, arr))), eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.maximum(np.diag(cov_alt), 0.0))
            self.standard_errors = ses
            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors
        return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Forecast h_{t} forward, keeping x_{t} fixed at last observed value.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Array of shape (horizon,) of variance forecasts.
        """
        if self.optimal_params is None: raise ValueError("Model must be optimized before forecasting.")

        omega = self.optimal_params["omega"]; beta = self.optimal_params["beta"]; gamma_ = self.optimal_params["gamma"]
        T = len(self.log_returns)
        h = np.zeros(T, dtype=float)
        h[0] = float(np.var(self.log_returns[:50]) if T >= 1 else 1.0)
        for t in range(1, T): h[t] = omega + beta * h[t - 1] + gamma_ * self.x[t - 1]
        prev = float(h[-1]); x_last = float(self.x[-1])

        forecasts: List[float] = []
        for _ in range(int(horizon)):
            nxt = float(omega + beta * prev + gamma_ * x_last)
            forecasts.append(nxt)
            prev = nxt
        return np.asarray(forecasts, dtype=float)
