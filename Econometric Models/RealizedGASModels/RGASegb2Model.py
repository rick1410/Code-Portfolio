from typing import Dict, Optional, Tuple, Union, List
import numpy as np
from scipy.special import digamma, polygamma, gamma
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.optimize import minimize, approx_fprime


class RGASegb2Model:
    """
    Realized GAS model with EGB2 return innovations and Gamma measurement on a realized kernel.

    State/score recursion
    ---------------------
    Let f_t be the log-variance and h_t = exp(f_t). The score splits into:
        nabla_t = (nu/2) * (X_t / h_t - 1) + A_t^EGB2(r_t, h_t; p, q),
    and the state update is:
        f_{t+1} = omega + beta * f_t + alpha * nabla_t.

    The return density is EGB2 with parameters (p, q), and the realized measure X_t is modeled
    via a Gamma likelihood conditional on h_t.

    Attributes
    ----------
    model_name : str
        Short model identifier.
    distribution : str
        Innovation distribution name.
    data : np.ndarray
        Array of returns r_t.
    realized_kernel : Optional[np.ndarray]
        Realized measure X_t used in the Gamma measurement and score.
    optimal_params : Optional[Dict[str, float]]
        Last optimized parameters.
    log_likelihood_value : Optional[float]
        Objective value at optimum (negative average log-likelihood with this implementation).
    aic : Optional[float]
        Akaike Information Criterion computed via `compute_aic_bic` (see method doc).
    bic : Optional[float]
        Bayesian Information Criterion computed via `compute_aic_bic` (see method doc).
    convergence : Optional[bool]
        Whether the last optimization converged.
    standard_errors : Optional[np.ndarray]
        Asymptotic standard errors from inverse Hessian (if computed).
    """

    model_name: str = "RGAS-EGB2"
    distribution: str = "EGB2"

    def __init__(self, data: np.ndarray, realized_kernel: Optional[np.ndarray] = None) -> None:
        """
        Initialize model containers.

        Parameters
        ----------
        data : np.ndarray
            Array of returns r_t.
        realized_kernel : Optional[np.ndarray], default None
            Realized measure X_t (same length as data).
        """
        self.data: np.ndarray = np.asarray(data, dtype=float).ravel()
        self.realized_kernel: Optional[np.ndarray] = None if realized_kernel is None else np.asarray(realized_kernel, dtype=float).ravel()
        self.optimal_params: Optional[Dict[str, float]] = None
        self.log_likelihood_value: Optional[float] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None
        self.standard_errors: Optional[np.ndarray] = None

    def log_likelihood(self, params: List[float]) -> float:
        """
        Negative average log-likelihood for Realized GAS with EGB2 returns and Gamma measurement.

        Parameters
        ----------
        params : List[float]
            [omega, alpha, beta, nu, p, q].

        Returns
        -------
        float
            Negative average log-likelihood.
        """
        omega, alpha, beta, nu, p, q = params
        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data[:50]))
        mu_t = 0.0

        Delta = digamma(p) - digamma(q)
        Omega = polygamma(1, p) + polygamma(1, q)

        for t in range(T - 1):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            f_t = f[t]
            h_t = np.exp(f_t)

            term1 = (nu / 2.0) * ((X_t / h_t) - 1.0)
            term2 = (np.sqrt(Omega) * (-q - p) * r_t * np.exp(np.sqrt(Omega) * r_t / np.sqrt(h_t) + Delta) / (2.0 * np.sqrt(h_t) * (np.exp(np.sqrt(Omega) * r_t / np.sqrt(h_t) + Delta) + 1.0)) - 0.5 - np.sqrt(Omega) * p * r_t / (2.0 * np.sqrt(h_t)))
            nabla_t = term1 + term2
            f[t + 1] = omega + beta * f_t + alpha * nabla_t

        ll = 0.0
        for t in range(T):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            f_t = f[t]
            h_t = np.exp(f_t)

            # Gamma measurement
            term_g1 = -np.log(gamma(nu / 2.0))
            term_g2 = -(nu / 2.0) * np.log(2.0 * h_t / nu)
            term_g3 = (nu / 2.0 - 1.0) * np.log(X_t)
            term_g4 = -(nu * X_t) / (2.0 * h_t)

            # EGB2 returns
            term_r = (0.5 * np.log(Omega) + p * (np.sqrt(Omega) * (r_t - mu_t) / np.sqrt(h_t) + Delta) - 0.5 * np.log(h_t) - np.log(gamma(p) * gamma(q) / gamma(p + q)) - (p + q) * np.log(1.0 + np.exp(np.sqrt(Omega) * (r_t - mu_t) / np.sqrt(h_t) + Delta)))

            ll += term_g1 + term_g2 + term_g3 + term_g4 + term_r

        return -float(ll) / T

    def compute_aic_bic(self, optimal_params: List[float]) -> Tuple[float, float]:
        """
        Compute AIC and BIC based on the EGB2 return contribution (as in original implementation).

        Notes
        -----
        This routine reconstructs only the EGB2 return log-likelihood contribution (without the Gamma part)
        to match the original behavior.

        Parameters
        ----------
        optimal_params : List[float]
            [omega, alpha, beta, nu, p, q] at optimum.

        Returns
        -------
        Tuple[float, float]
            (AIC, BIC).
        """
        omega, alpha, beta, nu, p, q = optimal_params
        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data))
        mu_t = 0.0

        Delta = digamma(p) - digamma(q)
        Omega = polygamma(1, p) + polygamma(1, q)

        for t in range(T - 1):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            f_t = f[t]
            h_t = np.exp(f_t)

            term1 = (nu / 2.0) * ((X_t / h_t) - 1.0)
            term2 = (np.sqrt(Omega) * (-q - p) * r_t * np.exp(np.sqrt(Omega) * r_t / np.sqrt(h_t) + Delta) / (2.0 * np.sqrt(h_t) * (np.exp(np.sqrt(Omega) * r_t / np.sqrt(h_t) + Delta) + 1.0)) - 0.5 - np.sqrt(Omega) * p * r_t / (2.0 * np.sqrt(h_t)))
            nabla_t = term1 + term2
            f[t + 1] = omega + beta * f_t + alpha * nabla_t

        ll = 0.0
        for t in range(T):
            r_t = self.data[t]
            f_t = f[t]
            h_t = np.exp(f_t)
            ll += (0.5 * np.log(Omega) + p * (np.sqrt(Omega) * (r_t - mu_t) / np.sqrt(h_t) + Delta) - 0.5 * np.log(h_t) - np.log(gamma(p) * gamma(q) / gamma(p + q)) - (p + q) * np.log(1.0 + np.exp(np.sqrt(Omega) * (r_t - mu_t) / np.sqrt(h_t) + Delta)))

        k = len(optimal_params)
        aic = 2.0 * k - 2.0 * ll
        bic = np.log(T) * k - 2.0 * ll
        print("The log-likelihood:", ll / T)
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Optimize parameters via SLSQP subject to bounds.

        Parameters
        ----------
        initial_params : Optional[Dict[str, float]], default None
            Initial parameter dict; if None and previous optimum exists, reuse it; else use defaults.
        compute_metrics : bool, default False
            If True, compute AIC/BIC and standard errors from the Hessian.

        Returns
        -------
        Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]
            If compute_metrics: (params, AIC, BIC, objective_value, std_errors); else params.
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {"omega": float(np.var(self.data[:50]) * (1.0 - 0.1 - 0.7)), "alpha": 0.1, "beta": 0.7, "nu": 3.0, "p": 3.5, "q": 3.5}

        keys: List[str] = list(initial_params.keys())
        x0: List[float] = list(initial_params.values())

        bounds = [(-100, 100), (0, 10), (0, 10), (0.001, 100), (0.001, 100), (0.001, 100)]

        result = minimize(self.log_likelihood, x0, method="SLSQP", bounds=bounds)

        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in result.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")

        if compute_metrics and self.convergence:
            self.log_likelihood_value = -float(result.fun)
            self.aic, self.bic = self.compute_aic_bic(result.x)

            hessian = approx_hess1(list(self.optimal_params.values()), self.log_likelihood, args=())
            hessian_inv = inv(hessian) / len(self.data)
            self.standard_errors = np.sqrt(np.maximum(np.diag(hessian_inv), 0.0))

            if np.isnan(self.standard_errors).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = approx_fprime(list(self.optimal_params.values()), self.log_likelihood, eps)
                h_alt = np.outer(grad, grad)
                self.standard_errors = np.sqrt(np.maximum(np.diag(h_alt), 0.0))

            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors
        else:
            return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Forecast h_{T+1}, …, h_{T+horizon} on the variance scale with zero score beyond step 1.

        After the first step (computed with the last observed score), future f's evolve as:
            f_{t+1} = omega + beta * f_t.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecasted variances.
        """
        if self.optimal_params is None:
            raise RuntimeError("Model must be optimized before forecasting.")

        omega, alpha, beta, nu, p, q = list(self.optimal_params.values())
        Delta = digamma(p) - digamma(q)
        Omega = polygamma(1, p) + polygamma(1, q)

        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data[:50]))
        for t in range(1, T):
            r_tm1 = self.data[t - 1]
            X_tm1 = self.realized_kernel[t - 1]
            h_tm1 = np.exp(f[t - 1])

            term1 = (nu / 2.0) * ((X_tm1 / h_tm1) - 1.0)
            term2 = (np.sqrt(Omega) * (-q - p) * r_tm1 * np.exp(np.sqrt(Omega) * r_tm1 / np.sqrt(h_tm1) + Delta) / (2.0 * np.sqrt(h_tm1) * (np.exp(np.sqrt(Omega) * r_tm1 / np.sqrt(h_tm1) + Delta) + 1.0)) - 0.5 - np.sqrt(Omega) * p * r_tm1 / (2.0 * np.sqrt(h_tm1)))
            f[t] = omega + beta * f[t - 1] + alpha * (term1 + term2)

        r_T = self.data[-1]
        X_T = self.realized_kernel[-1]
        f_T = f[-1]
        h_T = np.exp(f_T)

        term1 = (nu / 2.0) * ((X_T / h_T) - 1.0)
        term2 = (np.sqrt(Omega) * (-q - p) * r_T * np.exp(np.sqrt(Omega) * r_T / np.sqrt(h_T) + Delta) / (2.0 * np.sqrt(h_T) * (np.exp(np.sqrt(Omega) * r_T / np.sqrt(h_T) + Delta) + 1.0)) - 0.5 - np.sqrt(Omega) * p * r_T / (2.0 * np.sqrt(h_T)))
        f1 = omega + beta * f_T + alpha * (term1 + term2)

        forecasts: List[float] = [float(np.exp(f1))]
        prev = f1
        for _ in range(1, horizon):
            prev = omega + beta * prev
            forecasts.append(float(np.exp(prev)))

        return np.asarray(forecasts, dtype=float)
