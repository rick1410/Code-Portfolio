from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.special import gamma, digamma, polygamma
import scipy
from statsmodels.tools.numdiff import approx_hess1
from scipy.linalg import inv


class RGAStModel:
    """
    Realized GAS-t model with Student's t return innovations and Gamma measurement for a realized kernel.

    The latent log-variance state :math:`f_t` evolves via a GAS(1,1)-type recursion:
        :math:`f_{t+1} = \\omega + \\beta f_t + \\alpha \\nabla_t`,
    where the score :math:`\\nabla_t` combines a Gamma measurement component from the realized kernel
    and the Student's t (with :math:`\\nu_1`) return component.

    Attributes
    ----------
    model_name : str
        Short model identifier.
    distribution : str
        Innovation distribution name.
    data : np.ndarray
        Array of returns :math:`r_t`.
    realized_kernel : np.ndarray
        Realized measure :math:`X_t` used in the Gamma measurement (same length as `data`).
    optimal_params : Optional[Dict[str, float]]
        Last optimized parameters if available.
    standard_errors : Optional[np.ndarray]
        Asymptotic standard errors from inverse Hessian (if computed).
    log_likelihood_value : Optional[float]
        Objective value at optimum (negative average log-likelihood).
    aic : Optional[float]
        Akaike Information Criterion computed in `optimize(..., compute_metrics=True)`.
    bic : Optional[float]
        Bayesian Information Criterion computed in `optimize(..., compute_metrics=True)`.
    convergence : Optional[bool]
        Whether the last optimization converged.
    """

    model_name: str = "RGAS-t"
    distribution: str = "Student t"

    def __init__(self, data: np.ndarray, realized_kernel: np.ndarray) -> None:
        """
        Initialize containers and store input series.

        Parameters
        ----------
        data : np.ndarray
            Array of returns :math:`r_t`.
        realized_kernel : np.ndarray
            Realized measure :math:`X_t` (same length as `data`).
        """
        self.data: np.ndarray = np.asarray(data, dtype=float).ravel()
        self.realized_kernel: np.ndarray = np.asarray(realized_kernel, dtype=float).ravel()
        self.optimal_params: Optional[Dict[str, float]] = None
        self.standard_errors: Optional[np.ndarray] = None
        self.log_likelihood_value: Optional[float] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None

    def log_likelihood(self, params: List[float]) -> float:
        """
        Negative average log-likelihood for RGAS with Student's t returns and Gamma measurement.

        Parameters
        ----------
        params : List[float]
            Parameter vector [omega, alpha, beta, nu, nu1] with nu>0 and nu1>2.

        Returns
        -------
        float
            Negative average log-likelihood.
        """
        omega, alpha, beta, nu, nu1 = params
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
            term3 = ((nu1 + 1.0) / 2.0) * (r_t - mu_t) ** 2 / ((nu1 - 2.0) * np.exp(f_t) + (r_t - mu_t) ** 2)
            nabla_t = term1 + term2 + term3
            f[t + 1] = omega + beta * f_t + alpha * nabla_t

        ll = 0.0
        for t in range(T):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            f_t = f[t]
            g1 = -np.log(gamma(nu / 2.0))
            g2 = -(nu / 2.0) * np.log(2.0 * np.exp(f_t) / nu)
            g3 = (nu / 2.0 - 1.0) * np.log(X_t)
            g4 = -(nu * X_t) / (2.0 * np.exp(f_t))
            t1 = np.log(gamma((nu1 + 1.0) / 2.0)) - np.log(gamma(nu1 / 2.0))
            t2 = -0.5 * np.log((nu1 - 2.0) * np.pi * np.exp(f_t))
            t3 = -((nu1 + 1.0) / 2.0) * np.log(1.0 + (r_t - mu_t) ** 2 / ((nu1 - 2.0) * np.exp(f_t)))
            ll += g1 + g2 + g3 + g4 + t1 + t2 + t3

        return -float(ll) / T

    def compute_aic_bic(self, optimal_params: List[float]) -> Tuple[float, float]:
        """
        Compute AIC and BIC using only the Student's t return contribution (original behavior).

        Parameters
        ----------
        optimal_params : List[float]
            [omega, alpha, beta, nu, nu1] at optimum.

        Returns
        -------
        Tuple[float, float]
            (AIC, BIC).
        """
        omega, alpha, beta, nu, nu1 = optimal_params
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
            term3 = ((nu1 + 1.0) / 2.0) * (r_t - mu_t) ** 2 / ((nu1 - 2.0) * np.exp(f_t) + (r_t - mu_t) ** 2)
            nabla_t = term1 + term2 + term3
            f[t + 1] = omega + beta * f_t + alpha * nabla_t

        ll = 0.0
        for t in range(T):
            r_t = self.data[t]
            f_t = f[t]
            t1 = np.log(gamma((nu1 + 1.0) / 2.0)) - np.log(gamma(nu1 / 2.0))
            t2 = -0.5 * np.log((nu1 - 2.0) * np.pi * np.exp(f_t))
            t3 = -((nu1 + 1.0) / 2.0) * np.log(1.0 + (r_t - mu_t) ** 2 / ((nu1 - 2.0) * np.exp(f_t)))
            ll += t1 + t2 + t3

        k = len(optimal_params)
        aic = 2.0 * k - 2.0 * ll
        bic = np.log(T) * k - 2.0 * ll
        print("The log-likelihood:", ll / T)
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Optimize parameters via SLSQP subject to bounds and a stability constraint.

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
                initial_params = {"omega": float(np.var(self.data[:50]) * (1.0 - 0.1 - 0.7)), "alpha": 0.1, "beta": 0.7, "nu": 3.0, "nu1": 3.0}

        keys: List[str] = list(initial_params.keys())
        x0: List[float] = list(initial_params.values())

        bounds = [(-100, 100), (0, 10), (0, 10), (0.001, 100), (2.000001, 100)]
        constraints = {"type": "ineq", "fun": lambda x: 1 - x[2]}
        result = scipy.optimize.minimize(self.log_likelihood, x0, method="SLSQP", bounds=bounds, constraints=constraints)

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
        as an AR(1) in :math:`f`.

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

        omega, alpha, beta, nu, nu1 = list(self.optimal_params.values())

        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data[:50]))
        for t in range(1, T):
            r_t = self.data[t - 1]
            X_t = self.realized_kernel[t - 1]
            f_t = f[t - 1]
            term1 = (nu / 2.0) * (X_t / np.exp(f_t) - 1.0)
            term2 = -0.5
            term3 = ((nu1 + 1.0) / 2.0) * (r_t ** 2) / ((nu1 - 2.0) * np.exp(f_t) + r_t ** 2)
            nabla_t = term1 + term2 + term3
            f[t] = omega + beta * f_t + alpha * nabla_t

        f_last = f[-1]
        forecasts: List[float] = []
        for _ in range(horizon):
            f_last = omega + beta * f_last
            forecasts.append(float(np.exp(f_last)))

        return np.asarray(forecasts, dtype=float)
