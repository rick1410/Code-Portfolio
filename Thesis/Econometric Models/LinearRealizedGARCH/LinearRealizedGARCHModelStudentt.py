from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.special import gammaln


class LinearRealizedGARCHModelStudentt:
    """
    Linear Realized GARCH(1,1) with Student's t innovations for returns and a Gaussian
    measurement equation for the realized measure.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of log-returns.
    x : np.ndarray
        1-D array of realized measures aligned with `log_returns`.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Error distribution name.
    log_returns : np.ndarray
        Return series.
    x : np.ndarray
        Realized measure series.
    optimal_params : Optional[Dict[str, float]]
        Last fitted parameter set (in *transformed* space for nu/omega/... where specified
        by optimize; accessible as returned dict from `optimize`).
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (sum over observations, not averaged).
    aic : Optional[float]
        Akaike Information Criterion.
    bic : Optional[float]
        Bayesian Information Criterion.
    convergence : Optional[bool]
        Optimizer success flag.
    standard_errors : Optional[np.ndarray]
        Standard errors from inverse Hessian (may be NaN if inversion fails).
    """

    model_name: str = "LRGARCH-t"
    distribution: str = "Student t"

    log_returns: np.ndarray
    x: np.ndarray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: Optional[bool]
    standard_errors: Optional[np.ndarray]

    def __init__(self, log_returns: np.ndarray, x: np.ndarray) -> None:
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = None
        self.standard_errors = None

    def log_likelihood(self, params: Dict[str, float]) -> float:
        """
        Average negative log-likelihood for the Linear Realized GARCH(1,1) with Student's t errors.

        Parameters
        ----------
        params : dict
            Dictionary of parameters in transformed space:
            {'omega','beta','gamma','xi','phi','tau_1','tau_2','sigma_u','nu1'},
            where strictly-positive parameters are exponentiated internally.

        Returns
        -------
        float
            Average negative log-likelihood (to minimize).
        """
        T = len(self.log_returns)
        omega = float(np.exp(params["omega"]))
        beta = float(np.exp(params["gamma"]) / (np.exp(params["gamma"]) + np.exp(params["beta"]) + 1.0))
        gamma_ = float(np.exp(-params["phi"]) * (np.exp(params["beta"]) / (np.exp(params["gamma"]) + np.exp(params["beta"]) + 1.0)))
        xi = float(np.exp(params["xi"]))
        phi = float(np.exp(params["phi"]))
        tau_1 = float(params["tau_1"])
        tau_2 = float(params["tau_2"])
        sigma_u = float(np.exp(params["sigma_u"]))
        nu = float(np.exp(params["nu1"]))

        if sigma_u <= 0.0 or nu <= 2.0: return 1e10

        h = np.zeros(T, dtype=float)
        z = np.zeros(T, dtype=float)
        tau = np.zeros(T, dtype=float)
        u = np.zeros(T, dtype=float)

        h[0] = float(np.var(self.log_returns[:50]) if T >= 1 else 1.0)
        for t in range(1, T): h[t] = omega + beta * h[t - 1] + gamma_ * self.x[t - 1]

        for t in range(T):
            if h[t] <= 0.0 or not np.isfinite(h[t]): return 1e10
            z[t] = self.log_returns[t] / np.sqrt(h[t])
            tau[t] = tau_1 * z[t] + tau_2 * (z[t] ** 2 - 1.0)
            u[t] = self.x[t] - xi - phi * h[t] - tau[t]

        loglik_r = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
                    - 0.5 * np.log((nu - 2.0) * np.pi * h)
                    - (nu + 1.0) / 2.0 * np.log(1.0 + (self.log_returns ** 2) / ((nu - 2.0) * h)))
        loglik_x = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma_u ** 2) + (u ** 2) / (sigma_u ** 2))
        return float(-np.mean(loglik_r + loglik_x))

    def compute_aic_bic(self, log_likelihood: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC from total log-likelihood.

        Parameters
        ----------
        log_likelihood : float
            Total log-likelihood (not averaged).
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        (float, float)
            AIC, BIC.
        """
        n = len(self.log_returns)
        aic = 2.0 * num_params - 2.0 * (log_likelihood * 1.0)
        bic = np.log(n) * num_params - 2.0 * (log_likelihood * 1.0)
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Estimate parameters by unconstrained BFGS in transformed space.

        Parameters
        ----------
        initial_params : dict or None, default None
            Starting values in transformed space; if None, uses defaults.
        compute_metrics : bool, default False
            If True, also return (AIC, BIC, total_LL, SEs).

        Returns
        -------
        dict or tuple
            Params dict (mapped back to natural scale);
            or (params, AIC, BIC, total_LL, SEs) if `compute_metrics=True`.
        """
        if initial_params is None:
            initial_params = {
                "omega": float(np.log(np.exp(0.06))),
                "beta": float(np.log(0.41 * 1.04 / (1.0 - 0.41 * 1.04 - 0.55))),
                "gamma": float(np.log(0.55 / (1.0 - 0.41 * 1.04 - 0.55))),
                "xi": float(np.log(np.exp(-0.18))),
                "phi": float(np.log(1.04)),
                "tau_1": -0.07,
                "tau_2": 0.07,
                "sigma_u": float(np.log(0.38)),
                "nu1": float(np.log(5.0)),
            }
            first_window = True
        else:
            first_window = False

        keys = list(initial_params.keys())
        x0 = [float(v) for v in initial_params.values()]
        obj = lambda arr: self.log_likelihood(dict(zip(keys, arr)))
        results = minimize(obj, x0, method="BFGS")
        self.convergence = bool(results.success)
        fallback = initial_params.copy()

        if self.convergence:
            self.optimal_params = {
                "omega": float(np.exp(results.x[keys.index("omega")])),
                "beta": float(np.exp(results.x[keys.index("gamma")]) / (np.exp(results.x[keys.index("beta")]) + np.exp(results.x[keys.index("gamma")]) + 1.0)),
                "gamma": float(np.exp(-results.x[keys.index("phi")]) * (np.exp(results.x[keys.index("beta")]) / (np.exp(results.x[keys.index("beta")]) + np.exp(results.x[keys.index("gamma")]) + 1.0))),
                "xi": float(np.exp(results.x[keys.index("xi")])),
                "phi": float(np.exp(results.x[keys.index("phi")])),
                "tau_1": float(results.x[keys.index("tau_1")]),
                "tau_2": float(results.x[keys.index("tau_2")]),
                "sigma_u": float(np.exp(results.x[keys.index("sigma_u")])),
                "nu1": float(np.exp(results.x[keys.index("nu1")])),
            }
            print(f"Model: {self.model_name} | Convergence: Success")
        elif first_window:
            self.optimal_params = fallback
            print(f"Warning: First optimization failed for {self.model_name}. Using initial parameters.")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. No update to parameters.")
            return self.optimal_params if self.optimal_params is not None else fallback

        if compute_metrics and self.convergence:
            # Here results.fun is the average negative log-likelihood; convert to total LL
            self.log_likelihood_value = float(-results.fun * len(self.log_returns))
            num_params = len(self.optimal_params)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, num_params)

            # Standard errors via numerical Hessian (on natural-scale params as in original)
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
        Forecast conditional variance h_t forward keeping the realized measure fixed at its last value.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Array of length `horizon` with variance forecasts.
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
