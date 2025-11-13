from typing import Dict, Optional, Tuple, Union, List
import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.special import gamma
import scipy


class RGASModel:
    """
    Realized GAS model with Gaussian return innovations and Gamma measurement for a realized kernel.

    The latent log-variance state :math:`f_t` evolves according to a GAS(1,1)-type recursion:
        :math:`f_{t+1} = \\omega + \\beta f_t + \\alpha \\nabla_t`,
    where the score :math:`\\nabla_t` combines a Gamma measurement component from the realized kernel
    and the Gaussian return component:
        :math:`\\nabla_t = (\\nu/2)(X_t/\\exp(f_t) - 1) - 1/2 + (r_t - \\mu_t)^2/(2\\exp(f_t))`.

    Attributes
    ----------
    model_name : str
        Short model identifier.
    distribution : str
        Innovation distribution name.
    data : np.ndarray
        Array of returns :math:`r_t`.
    realized_kernel : Optional[np.ndarray]
        Realized measure :math:`X_t` used in the Gamma measurement (same length as `data`).
    optimal_params : Optional[Dict[str, float]]
        Last optimized parameters if available.
    log_likelihood_value : Optional[float]
        Objective value at optimum (negative average log-likelihood).
    aic : Optional[float]
        Akaike Information Criterion computed in `optimize(..., compute_metrics=True)`.
    bic : Optional[float]
        Bayesian Information Criterion computed in `optimize(..., compute_metrics=True)`.
    convergence : Optional[bool]
        Whether the last optimization converged.
    standard_errors : Optional[np.ndarray]
        Asymptotic standard errors from inverse Hessian (if computed).
    """

    model_name: str = "RGAS"
    distribution: str = "Normal"

    def __init__(self, data: np.ndarray, realized_kernel: Optional[np.ndarray] = None) -> None:
        """
        Initialize containers and store input series.

        Parameters
        ----------
        data : np.ndarray
            Array of returns :math:`r_t`.
        realized_kernel : Optional[np.ndarray], default None
            Realized measure :math:`X_t` (same length as `data`).
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
        Negative average log-likelihood for RGAS with Gaussian returns and Gamma measurement.

        Parameters
        ----------
        params : List[float]
            Parameter vector [omega, alpha, beta, nu].

        Returns
        -------
        float
            Negative average log-likelihood.
        """
        omega, alpha, beta, nu = params
        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data[:50]))
        mu_t = 0.0

        for t in range(T - 1):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            f_t = f[t]
            term1 = (nu / 2.0) * (X_t / np.exp(f_t) - 1.0)
            term2 = -0.5
            term3 = (r_t - mu_t) ** 2 / (2.0 * np.exp(f_t))
            nabla_t = term1 + term2 + term3
            f[t + 1] = omega + beta * f_t + alpha * nabla_t

        ll = 0.0
        for t in range(T):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            f_t = f[t]
            # Gamma measurement
            g1 = -np.log(gamma(nu / 2.0))
            g2 = -(nu / 2.0) * np.log(2.0 * np.exp(f_t) / nu)
            g3 = (nu / 2.0 - 1.0) * np.log(X_t)
            g4 = -(nu * X_t) / (2.0 * np.exp(f_t))
            # Gaussian return
            rll = -0.5 * (np.log(2.0 * np.pi) + f_t + (r_t - mu_t) ** 2 / np.exp(f_t))
            ll += g1 + g2 + g3 + g4 + rll

        return -float(ll) / T

    def compute_aic_bic(self, optimal_params: List[float]) -> Tuple[float, float]:
        """
        Compute AIC and BIC using only the Gaussian return contribution (original behavior).

        Parameters
        ----------
        optimal_params : List[float]
            [omega, alpha, beta, nu] at optimum.

        Returns
        -------
        Tuple[float, float]
            (AIC, BIC).
        """
        omega, alpha, beta, nu = optimal_params
        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data))
        mu_t = 0.0

        for t in range(T - 1):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            f_t = f[t]
            term1 = (nu / 2.0) * (X_t / np.exp(f_t) - 1.0)
            term2 = -0.5
            term3 = (r_t - mu_t) ** 2 / (2.0 * np.exp(f_t))
            nabla_t = term1 + term2 + term3
            f[t + 1] = omega + beta * f_t + alpha * nabla_t

        ll = 0.0
        for t in range(T):
            r_t = self.data[t]
            f_t = f[t]
            ll += -0.5 * (np.log(2.0 * np.pi) + f_t + (r_t - mu_t) ** 2 / np.exp(f_t))

        k = len(optimal_params)
        aic = 2.0 * k - 2.0 * ll
        bic = np.log(T) * k - 2.0 * ll
        print("The log-likelihood:", ll / T)
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Optimize parameters via SLSQP subject to bounds and a simple stability constraint.

        Parameters
        ----------
        initial_params : Optional[Dict[str, float]], default None
            Initial parameter dict; if None, reuse previous optimum if present, else defaults.
        compute_metrics : bool, default False
            If True, compute AIC/BIC and asymptotic standard errors.

        Returns
        -------
        Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]
            If `compute_metrics`: (params, AIC, BIC, objective_value, std_errors); else params.
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {"omega": float(np.var(self.data[:50]) * (1.0 - 0.1 - 0.7)), "alpha": 0.1, "beta": 0.7, "nu": 3.0}

        keys: List[str] = list(initial_params.keys())
        x0: List[float] = list(initial_params.values())

        bounds = [(-100, 100), (0, 10), (0, 10), (0.001, 100)]
        constraints = {"type": "ineq", "fun": lambda x: 2 - np.abs(x[2])}
        result = minimize(self.log_likelihood, x0, method="SLSQP", bounds=bounds, constraints=constraints)

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
                grad = scipy.optimize.approx_fprime(list(self.optimal_params.values()), self.log_likelihood, eps)
                h_alt = np.outer(grad, grad)
                self.standard_errors = np.sqrt(np.maximum(np.diag(h_alt), 0.0))

            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors
        else:
            return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Forecast :math:`h_{T+1},\\ldots,h_{T+\\text{horizon}}` on the variance scale with zero score beyond the sample.

        After rebuilding the in-sample state up to :math:`T`, the forecast sets the score to zero and evolves
        as an AR(1) in :math:`f` for steps :math:`\\ge 2`.

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

        omega, alpha, beta, nu = list(self.optimal_params.values())

        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data[:50]))
        for t in range(1, T):
            r_tm1 = self.data[t - 1]
            X_tm1 = self.realized_kernel[t - 1]
            f_tm1 = f[t - 1]
            term1 = (nu / 2.0) * (X_tm1 / np.exp(f_tm1) - 1.0)
            term2 = -0.5
            term3 = (r_tm1 ** 2) / (2.0 * np.exp(f_tm1))
            nabla = term1 + term2 + term3
            f[t] = omega + beta * f_tm1 + alpha * nabla

        f_prev = f[-1]

        forecasts: List[float] = [float(np.exp(f_prev))]
        for _ in range(1, horizon):
            f_prev = omega + beta * f_prev
            forecasts.append(float(np.exp(f_prev)))

        return np.asarray(forecasts, dtype=float)
