from typing import Dict, Optional, Tuple, Union, List
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.optimize import approx_fprime


class RealizedEGARCHNormalModel:
    """
    Realized EGARCH(1,1) with Gaussian innovations and Gaussian measurement for the realized proxy.

    The state (log-variance) evolves as:
        log h_t = omega + beta * log h_{t-1} + tau_{t-1} + delta * u_{t-1},
    where tau_t = tau_1 * z_t + tau_2 * (z_t^2 - 1), z_t = r_t / sqrt(h_t),
    and u_t = log(x_t) - xi - log h_t.

    Attributes
    ----------
    model_name : str
        Short model identifier.
    distribution : str
        Name of the return innovation distribution.
    log_returns : np.ndarray
        Array of log returns r_t.
    x : np.ndarray
        Array of realized measures x_t (must be positive to take logs).
    optimal_params : Optional[Dict[str, float]]
        Dictionary of last optimized parameters on the *natural* scale.
    log_likelihood_value : Optional[float]
        Total (not average) log-likelihood at the optimum (if computed).
    aic : Optional[float]
        Akaike Information Criterion based on total log-likelihood (if computed).
    bic : Optional[float]
        Bayesian Information Criterion based on total log-likelihood (if computed).
    convergence : Optional[bool]
        Whether the last optimization succeeded.
    standard_errors : Optional[np.ndarray]
        Asymptotic standard errors from the inverse Hessian (if computed).
    """

    model_name: str = "REGARCH"
    distribution: str = "Normal"

    def __init__(self, log_returns: np.ndarray, x: np.ndarray) -> None:
        """
        Initialize the model with returns and realized measures.

        Parameters
        ----------
        log_returns : np.ndarray
            Array of log returns.
        x : np.ndarray
            Array of realized measures (e.g., volatility proxy).
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
        Negative average log-likelihood for Realized EGARCH(1,1) with Normal returns and measurement.

        Parameters
        ----------
        params : Dict[str, float]
            Parameter dict with keys: omega, beta, delta, xi, tau_1, tau_2, sigma_u.

        Returns
        -------
        float
            Negative average log-likelihood value.
        """
        T: int = len(self.log_returns)

        omega: float = params["omega"]
        beta: float = params["beta"]
        delta: float = params["delta"]
        xi: float = params["xi"]
        tau_1: float = params["tau_1"]
        tau_2: float = params["tau_2"]
        sigma_u: float = params["sigma_u"]

        log_h: np.ndarray = np.zeros(T)
        z: np.ndarray = np.zeros(T)
        u: np.ndarray = np.zeros(T)
        tau: np.ndarray = np.zeros(T)

        log_h[0] = np.log(np.var(self.log_returns[:50]))
        z[0] = self.log_returns[0] / np.sqrt(np.exp(log_h[0]))
        u[0] = np.log(self.x[0]) - xi - log_h[0]

        for t in range(1, T):
            tau[t - 1] = tau_1 * z[t - 1] + tau_2 * (z[t - 1] ** 2 - 1.0)
            log_h[t] = omega + beta * log_h[t - 1] + tau[t - 1] + delta * u[t - 1]
            z[t] = self.log_returns[t] / np.sqrt(np.exp(log_h[t]))
            u[t] = np.log(self.x[t]) - xi - log_h[t]

        loglik_r: np.ndarray = -0.5 * (np.log(2 * np.pi) + log_h + (self.log_returns ** 2) / np.exp(log_h))
        loglik_x: np.ndarray = -0.5 * (np.log(2 * np.pi) + np.log(sigma_u ** 2) + (u ** 2) / (sigma_u ** 2))
        return -float(np.mean(loglik_r + loglik_x))

    def calculate_aic_bic(self, log_likelihood: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC from total log-likelihood.

        Parameters
        ----------
        log_likelihood : float
            Total log-likelihood (not average).
        num_params : int
            Number of free parameters.

        Returns
        -------
        Tuple[float, float]
            (AIC, BIC).
        """
        aic: float = 2 * num_params - 2 * log_likelihood
        bic: float = np.log(len(self.log_returns)) * num_params - 2 * log_likelihood
        return aic, bic

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[None, Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Estimate the Realized EGARCH(1,1) using constrained optimization (Normal).

        Parameters
        ----------
        initial_params : Optional[Dict[str, float]], default None
            Initial parameter dictionary. If None and previous optimum exists, reuse it; else use defaults.
        compute_metrics : bool, default False
            If True and optimization converges, compute total log-likelihood, AIC/BIC, and standard errors.

        Returns
        -------
        Union[None, Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]
            If compute_metrics and converged: (params, AIC, BIC, total_loglik, std_errors);
            if converged without metrics: params;
            otherwise None (behavior preserved).
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {"omega": 0.06, "beta": 0.55, "delta": 0.41, "xi": -0.18, "tau_1": -0.07, "tau_2": 0.07, "sigma_u": 0.38}

        keys: List[str] = list(initial_params.keys())
        x0: List[float] = list(initial_params.values())

        result = minimize(lambda p: self.log_likelihood(dict(zip(keys, p))), x0, method="SLSQP", bounds=[(-np.inf, np.inf), (0, np.inf), (0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (1e-5, np.inf)], options={"eps": 1e-09})

        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in result.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")

        if compute_metrics and self.convergence:
            self.log_likelihood_value = -float(result.fun) * len(self.log_returns)
            num_params: int = len(self.optimal_params)
            self.aic, self.bic = self.calculate_aic_bic(self.log_likelihood_value, num_params)

            hessian = approx_hess1(list(self.optimal_params.values()), lambda p: self.log_likelihood(dict(zip(keys, p))), args=())
            hessian_inv = inv(hessian) / len(self.log_returns)
            self.standard_errors = np.sqrt(np.maximum(np.diag(hessian_inv), 0.0))

            if np.isnan(self.standard_errors).any():
                epsilon = float(np.sqrt(np.finfo(float).eps))
                grad = approx_fprime(list(self.optimal_params.values()), lambda p: self.log_likelihood(dict(zip(keys, p))), epsilon)
                hessian_alt = np.outer(grad, grad)
                self.standard_errors = np.sqrt(np.maximum(np.diag(hessian_alt), 0.0))

            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors
        else:
            self.optimal_params  # preserve original behavior (no return)

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Forecast h_{T+1}, …, h_{T+horizon} on the variance scale by freezing one-step-ahead shocks.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Array of forecasts for the conditional variance.
        """
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")

        omega: float = self.optimal_params["omega"]
        beta: float = self.optimal_params["beta"]
        delta: float = self.optimal_params["delta"]
        xi: float = self.optimal_params["xi"]
        tau_1: float = self.optimal_params["tau_1"]
        tau_2: float = self.optimal_params["tau_2"]

        T: int = len(self.log_returns)
        log_h: np.ndarray = np.zeros(T)
        z: np.ndarray = np.zeros(T)
        u: np.ndarray = np.zeros(T)
        tau: np.ndarray = np.zeros(T)

        log_h[0] = np.log(np.var(self.log_returns[:50]))
        z[0] = self.log_returns[0] / np.sqrt(np.exp(log_h[0]))
        u[0] = np.log(self.x[0]) - xi - log_h[0]

        for t in range(1, T):
            tau[t - 1] = tau_1 * z[t - 1] + tau_2 * (z[t - 1] ** 2 - 1.0)
            log_h[t] = omega + beta * log_h[t - 1] + tau[t - 1] + delta * u[t - 1]
            z[t] = self.log_returns[t] / np.sqrt(np.exp(log_h[t]))
            u[t] = np.log(self.x[t]) - xi - log_h[t]

        tau_last: float = tau_1 * z[-1] + tau_2 * (z[-1] ** 2 - 1.0)
        u_last: float = np.log(self.x[-1]) - xi - log_h[-1]
        prev_log_h: float = log_h[-1]

        forecasts: List[float] = []
        for _ in range(horizon):
            prev_log_h = omega + beta * prev_log_h + tau_last + delta * u_last
            forecasts.append(float(np.exp(prev_log_h)))

        return np.asarray(forecasts, dtype=float)
