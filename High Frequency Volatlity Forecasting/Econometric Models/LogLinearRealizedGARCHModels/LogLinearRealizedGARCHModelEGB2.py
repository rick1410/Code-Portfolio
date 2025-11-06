from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.special import gamma, digamma, polygamma
from scipy.optimize import approx_fprime


class LogLinearRealizedGARCHModelEGB2:
    """
    Log-linear Realized GARCH(1,1) with EGB2 innovations for returns and Gaussian
    measurement equation for the realized proxy.

    The state (log variance) evolves as:
        log h_t = omega + beta * log h_{t-1} + gamma1 * log x_{t-1},

    Returns follow an EGB2 density with parameters (p, q) and scaling by h_t,
    while the realized proxy obeys:
        log x_t = xi + phi * log h_t + tau(z_t) + u_t,   u_t ~ N(0, sigma_u^2),
    with leverage/measurement adjustment
        tau(z_t) = tau_1 * z_t + tau_2 * (z_t^2 - 1),   z_t = r_t / sqrt(h_t).

    Attributes
    ----------
    model_name : str
        Model identifier.
    distribution : str
        Return innovation family.
    log_returns : np.ndarray
        Array of returns r_t, shape (T,).
    x : np.ndarray
        Realized measure/proxy x_t, shape (T,).
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters when available.
    log_likelihood_value : Optional[float]
        Total log-likelihood at the optimum (not averaged).
    aic : Optional[float]
        Akaike Information Criterion (if computed).
    bic : Optional[float]
        Bayesian Information Criterion (if computed).
    convergence : Optional[bool]
        Optimizer convergence flag.
    standard_errors : Optional[np.ndarray]
        Asymptotic standard errors from inverse Hessian (if computed).
    """

    model_name: str = "LLRGARCH-EGB2"
    distribution: str = "EGB2"

    def __init__(self, log_returns: np.ndarray, x: np.ndarray) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        log_returns : np.ndarray
            Array of log returns r_t.
        x : np.ndarray
            Array of realized measures x_t (volatility proxy).
        """
        self.log_returns: np.ndarray = np.asarray(log_returns, dtype=float).ravel()
        self.x: np.ndarray = np.asarray(x, dtype=float).ravel()
        self.optimal_params: Optional[Dict[str, float]] = None
        self.log_likelihood_value: Optional[float] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None
        self.standard_errors: Optional[np.ndarray] = None

    def log_likelihood(self, params: Dict[str, float]) -> float:
        """
        Negative average log-likelihood for the log-linear Realized GARCH-EGB2.

        Parameters
        ----------
        params : Dict[str, float]
            Parameter dictionary with keys
            {'omega','beta','gamma1','xi','phi','tau_1','tau_2','sigma_u','p','q'}.

        Returns
        -------
        float
            Negative average log-likelihood to be minimized.
        """
        T = len(self.log_returns)
        omega = params["omega"]
        beta = params["beta"]
        gamma1 = params["gamma1"]
        xi = params["xi"]
        phi = params["phi"]
        tau_1 = params["tau_1"]
        tau_2 = params["tau_2"]
        sigma_u = params["sigma_u"]
        p = params["p"]
        q = params["q"]
        mu_t = 0.0

        log_h = np.zeros(T)
        z = np.zeros(T)
        u = np.zeros(T)
        tau = np.zeros(T)

        Delta = digamma(p) - digamma(q)
        Omega = polygamma(1, p) + polygamma(1, q)

        log_h[0] = np.log(np.var(self.log_returns[:50]))
        for t in range(1, T): log_h[t] = omega + beta * log_h[t - 1] + gamma1 * np.log(self.x[t - 1])
        for t in range(T):
            z[t] = self.log_returns[t] / np.sqrt(np.exp(log_h[t]))
            tau[t] = tau_1 * z[t] + tau_2 * (z[t] ** 2 - 1.0)
            u[t] = np.log(self.x[t]) - xi - phi * log_h[t] - tau[t]

        loglik_r = (
            0.5 * np.log(Omega)
            + p * (np.sqrt(Omega) * (self.log_returns - mu_t) / np.sqrt(np.exp(log_h)) + Delta)
            - 0.5 * log_h
            - np.log(gamma(p) * gamma(q) / gamma(p + q))
            - (p + q) * np.log(1.0 + np.exp(np.sqrt(Omega) * (self.log_returns - mu_t) / np.sqrt(np.exp(log_h)) + Delta))
        )
        loglik_x = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma_u**2) + (u**2) / (sigma_u**2))
        return float(-np.mean(loglik_r + loglik_x))

    def compute_aic_bic(self, log_likelihood: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC using the provided (total) log-likelihood.

        Parameters
        ----------
        log_likelihood : float
            Total log-likelihood value (not averaged).
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        Tuple[float, float]
            AIC and BIC values.
        """
        aic = 2.0 * num_params - 2.0 * log_likelihood
        bic = np.log(len(self.log_returns)) * num_params - 2.0 * log_likelihood
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Estimate parameters via SLSQP under stability constraint; optionally compute metrics.

        Parameters
        ----------
        initial_params : Optional[Dict[str, float]], default None
            If None, reuse last optimum when available; otherwise use preset initials.
        compute_metrics : bool, default False
            If True, return (params, AIC, BIC, loglik, std_errs); else return params.

        Returns
        -------
        Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]
            Parameter dictionary, or tuple with information criteria and standard errors.
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {
                    "omega": float(np.exp(0.06)),
                    "beta": 0.55,
                    "gamma1": 0.41,
                    "xi": float(np.exp(-0.18)),
                    "phi": 1.04,
                    "tau_1": -0.07,
                    "tau_2": 0.07,
                    "sigma_u": 0.38,
                    "p": 3.5,
                    "q": 3.5,
                }

        keys: List[str] = list(initial_params.keys())
        x0: List[float] = list(initial_params.values())
        bounds = [
            (-np.inf, np.inf), (0.0, np.inf), (0.0, np.inf), (-np.inf, np.inf),
            (0.0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (1e-5, np.inf),
            (1e-5, np.inf), (1e-5, np.inf),
        ]
        constraints = ({'type': 'ineq', 'fun': lambda x: 1 - x[1] - x[2] * x[4]})

        result = minimize(lambda arr: self.log_likelihood(dict(zip(keys, arr))), x0, method='SLSQP', bounds=bounds, constraints=constraints)
        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in result.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")

        if compute_metrics and self.convergence:
            self.log_likelihood_value = -float(result.fun) * len(self.log_returns)
            num_params = len(self.optimal_params)
            self.aic, self.bic = self.compute_aic_bic(float(self.log_likelihood_value), num_params)
            try:
                hessian = approx_hess1(np.asarray(list(self.optimal_params.values()), dtype=float), lambda arr: self.log_likelihood(dict(zip(keys, arr))))
                hessian_inv = inv(hessian) / max(len(self.log_returns), 1)
                self.standard_errors = np.sqrt(np.maximum(np.diag(hessian_inv), 0.0))
            except Exception:
                self.standard_errors = np.full(num_params, np.nan, dtype=float)

            if np.isnan(self.standard_errors).any():
                try:
                    eps = np.sqrt(np.finfo(float).eps)
                    grad = approx_fprime(np.asarray(list(self.optimal_params.values()), dtype=float), lambda arr: self.log_likelihood(dict(zip(keys, arr))), eps)
                    alt_inv = np.outer(grad, grad)
                    self.standard_errors = np.sqrt(np.maximum(np.diag(alt_inv), 0.0))
                except Exception:
                    self.standard_errors = np.full(num_params, np.nan, dtype=float)

            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors
        else:
            return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Multi-step-ahead forecasts for h_t assuming x_t is fixed at its last value.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecasted variances [h_{T+1}, …, h_{T+horizon}].
        """
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")

        omega = float(self.optimal_params["omega"])
        beta = float(self.optimal_params["beta"])
        gamma = float(self.optimal_params["gamma1"])

        T = len(self.log_returns)
        log_h = np.zeros(T)
        log_h[0] = np.log(np.var(self.log_returns[:50]))
        for t in range(1, T): log_h[t] = omega + beta * log_h[t - 1] + gamma * np.log(self.x[t - 1])

        log_h_last = float(log_h[-1])
        x_last = float(self.x[-1])

        forecasts: List[float] = []
        prev = log_h_last
        for _ in range(horizon):
            nxt = omega + beta * prev + gamma * np.log(x_last)
            forecasts.append(float(np.exp(nxt)))
            prev = nxt

        return np.asarray(forecasts, dtype=float)
