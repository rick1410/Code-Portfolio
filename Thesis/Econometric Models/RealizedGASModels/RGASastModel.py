from typing import Dict, Optional, Tuple, Union, List
import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.special import gamma


class RGASastModel:
    """
    Realized GAS model with Asymmetric Student-t (AST) return innovations and Gamma measurement on a realized kernel.

    State/score recursion
    ---------------------
    Let f_t be the log-variance and h_t = exp(f_t). The score splits into two parts:
        nabla_t = (nu/2) * (X_t / h_t - 1)  +  A_t(r_t, 0, h_t; delta, v1, v2),
    and the state update is
        f_{t+1} = omega + beta * f_t + alpha * nabla_t.

    The return density is AST with parameters (delta, v1, v2), and the realized measure X_t
    is modeled via a Gamma likelihood conditional on h_t.

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

    model_name: str = "RGAS-AST"
    distribution: str = "AST"

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

    def K(self, v: float) -> float:
        """
        K helper for AST distribution.

        Parameters
        ----------
        v : float
            Degrees of freedom.

        Returns
        -------
        float
            K(v) = Γ((v+1)/2) / (sqrt(pi*v) Γ(v/2)).
        """
        return float(gamma((v + 1.0) / 2.0) / (np.sqrt(np.pi * v) * gamma(v / 2.0)))

    def B(self, delta: float, v1: float, v2: float) -> float:
        """
        Normalizing B(delta, v1, v2) for AST.

        Parameters
        ----------
        delta : float
            Skewness weight in (0,1).
        v1 : float
            Left tail degrees of freedom (>2).
        v2 : float
            Right tail degrees of freedom (>2).

        Returns
        -------
        float
            B(delta, v1, v2).
        """
        return float(delta * self.K(v1) + (1.0 - delta) * self.K(v2))

    def alpha_star(self, delta: float, v1: float, v2: float) -> float:
        """
        AST asymmetry scaling α*.

        Parameters
        ----------
        delta : float
        v1 : float
        v2 : float

        Returns
        -------
        float
            α*(delta, v1, v2).
        """
        b = self.B(delta, v1, v2)
        return float((delta * self.K(v1)) / b)

    def m(self, delta: float, v1: float, v2: float) -> float:
        """
        AST location adjustment m.

        Parameters
        ----------
        delta : float
        v1 : float
        v2 : float

        Returns
        -------
        float
            m(delta, v1, v2).
        """
        a = self.alpha_star(delta, v1, v2)
        b = self.B(delta, v1, v2)
        return float(4.0 * b * (-a ** 2 * v1 / (v1 - 1.0) + (1.0 - a) ** 2 * v2 / (v2 - 1.0)))

    def s(self, delta: float, v1: float, v2: float) -> float:
        """
        AST scale adjustment s (>0).

        Parameters
        ----------
        delta : float
        v1 : float
        v2 : float

        Returns
        -------
        float
            s(delta, v1, v2) with numerical safeguard inside sqrt.
        """
        a = self.alpha_star(delta, v1, v2)
        m_val = self.m(delta, v1, v2)
        inside = 4.0 * (delta * a ** 2 * v1 / (v1 - 2.0) + (1.0 - delta) * (1.0 - a) ** 2 * v2 / (v2 - 2.0)) - m_val ** 2
        return float(np.sqrt(inside))

    def I_t(self, r_t: float, mu_t: float, h_t: float, m_val: float, s_val: float) -> int:
        """
        AST indicator I{ m + s * (r-mu)/sqrt(h) > 0 }.

        Parameters
        ----------
        r_t : float
        mu_t : float
        h_t : float
        m_val : float
        s_val : float

        Returns
        -------
        int
            1 if the argument is positive, else 0.
        """
        return 1 if m_val + s_val * (r_t - mu_t) / np.sqrt(h_t) > 0.0 else 0

    def A_t(self, r_t: float, mu_t: float, h_t: float, delta: float, v1: float, v2: float) -> float:
        """
        AST score component for returns used in GAS update.

        Parameters
        ----------
        r_t : float
        mu_t : float
        h_t : float
        delta : float
        v1 : float
        v2 : float

        Returns
        -------
        float
            Score contribution A_t.
        """
        m_val = self.m(delta, v1, v2)
        s_val = self.s(delta, v1, v2)
        a_star = self.alpha_star(delta, v1, v2)
        I = 1 if m_val + s_val * (r_t - mu_t) / np.sqrt(h_t) > 0.0 else 0
        num = s_val * (r_t - mu_t) / np.sqrt(h_t) + m_val
        t1 = num / (4.0 * (1.0 - a_star) ** 2 * v2 * (num ** 2 / (4.0 * (1.0 - a_star) ** 2 * v2) + 1.0) * np.sqrt(h_t))
        t2 = num / (4.0 * a_star ** 2 * v1 * (num ** 2 / (4.0 * a_star ** 2 * v1) + 1.0) * np.sqrt(h_t))
        return float(-0.5 + I * (v2 + 1.0) / 2.0 * t1 + (1.0 - I) * (v2 + 1.0) / 2.0 * t2)

    def log_likelihood(self, params: List[float]) -> float:
        """
        Negative average log-likelihood for Realized GAS-AST with Gamma measurement.

        Parameters
        ----------
        params : List[float]
            [omega, alpha, beta, delta, nu, v1, v2].

        Returns
        -------
        float
            Negative average log-likelihood.
        """
        omega, alpha, beta, delta, nu, v1, v2 = params
        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data[:50]))
        mu_t = 0.0

        for t in range(T - 1):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            h_t = np.exp(f[t])
            term1 = (nu / 2.0) * ((X_t / h_t) - 1.0)
            term2 = self.A_t(r_t, mu_t, h_t, delta, v1, v2)
            f[t + 1] = omega + beta * f[t] + alpha * (term1 + term2)

        ll = 0.0
        for t in range(T):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            f_t = f[t]
            h_t = np.exp(f_t)
            m_val = self.m(delta, v1, v2)
            s_val = self.s(delta, v1, v2)
            a_star = self.alpha_star(delta, v1, v2)
            I = 1 if m_val + s_val * (r_t - mu_t) / np.sqrt(h_t) > 0.0 else 0
            Bv = self.B(delta, v1, v2)

            log_gamma = (-np.log(gamma(nu / 2.0)) - (nu / 2.0) * np.log(2.0 * np.exp(f_t) / nu) + (nu / 2.0 - 1.0) * np.log(X_t) - (nu * X_t) / (2.0 * np.exp(f_t)))
            term5 = ((v1 + 1.0) / 2.0) * np.log(1.0 + (1.0 / v1) * ((m_val + s_val * (r_t - mu_t) / np.sqrt(h_t)) / (2.0 * a_star)) ** 2)
            term6 = ((v2 + 1.0) / 2.0) * np.log(1.0 + (1.0 / v2) * ((m_val + s_val * (r_t - mu_t) / np.sqrt(h_t)) / (2.0 * (1.0 - a_star))) ** 2)
            log_ast = (np.log(s_val) + np.log(Bv) - 0.5 * np.log(h_t) - (1 - I) * term5 - I * term6)

            ll += log_gamma + log_ast

        return -float(ll) / T

    def compute_aic_bic(self, optimal_params: List[float]) -> Tuple[float, float]:
        """
        Compute AIC and BIC as implemented in the original method (AST part only).

        Notes
        -----
        This routine reconstructs the AST log-likelihood contribution (without the Gamma part)
        exactly as in the original implementation and uses it to form AIC/BIC.

        Parameters
        ----------
        optimal_params : List[float]
            [omega, alpha, beta, delta, nu, v1, v2] at optimum.

        Returns
        -------
        Tuple[float, float]
            (AIC, BIC).
        """
        omega, alpha, beta, delta, nu, v1, v2 = optimal_params
        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data))
        mu_t = 0.0
        ll = 0.0

        for t in range(T - 1):
            r_t = self.data[t]
            X_t = self.realized_kernel[t]
            h_t = np.exp(f[t])
            term1 = (nu / 2.0) * ((X_t / h_t) - 1.0)
            term2 = self.A_t(r_t, mu_t, h_t, delta, v1, v2)
            f[t + 1] = omega + beta * f[t] + alpha * (term1 + term2)

        for t in range(T):
            h_t = np.exp(f[t])
            m_val = self.m(delta, v1, v2)
            s_val = self.s(delta, v1, v2)
            a_star = self.alpha_star(delta, v1, v2)
            I = self.I_t(self.data[t], mu_t, h_t, m_val, s_val)
            Bv = self.B(delta, v1, v2)
            term5 = ((v1 + 1.0) / 2.0) * np.log(1.0 + (1.0 / v1) * ((m_val + s_val * (self.data[t] - mu_t) / np.sqrt(h_t)) / (2.0 * a_star)) ** 2)
            term6 = ((v2 + 1.0) / 2.0) * np.log(1.0 + (1.0 / v2) * ((m_val + s_val * (self.data[t] - mu_t) / np.sqrt(h_t)) / (2.0 * (1.0 - a_star))) ** 2)
            ll += (np.log(s_val) + np.log(Bv) - 0.5 * np.log(h_t) - (1 - I) * term5 - I * term6)

        k = len(optimal_params)
        aic = 2.0 * k - 2.0 * ll
        bic = np.log(T) * k - 2.0 * ll
        print("The log-likelihood:", ll / T)
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Optimize parameters via SLSQP subject to bounds/constraint.

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
                initial_params = {"omega": 0.203, "alpha": 0.093, "beta": 0.91, "delta": 0.5, "nu": 8.8, "v1": 6.8, "v2": 6.8}

        keys: List[str] = list(initial_params.keys())
        x0: List[float] = list(initial_params.values())

        bounds = [(-100, 100), (0, 10), (0, 10), (0, 1), (0.001, 100), (0.001, 100), (0.001, 100)]
        constraints = {"type": "ineq", "fun": lambda x: 2.0 - np.abs(x[2])}

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

        omega, alpha, beta, delta, nu, v1, v2 = list(self.optimal_params.values())

        T = len(self.data)
        f = np.zeros(T)
        f[0] = np.log(np.var(self.data[:50]))
        for t in range(1, T):
            r_tm1 = self.data[t - 1]
            X_tm1 = self.realized_kernel[t - 1]
            h_tm1 = np.exp(f[t - 1])
            term1 = (nu / 2.0) * ((X_tm1 / h_tm1) - 1.0)
            term2 = self.A_t(r_tm1, 0.0, h_tm1, delta, v1, v2)
            f[t] = omega + beta * f[t - 1] + alpha * (term1 + term2)

        r_T = self.data[-1]
        X_T = self.realized_kernel[-1]
        f_T = f[-1]
        h_T = np.exp(f_T)
        term1 = (nu / 2.0) * ((X_T / h_T) - 1.0)
        term2 = self.A_t(r_T, 0.0, h_T, delta, v1, v2)
        f1 = omega + beta * f_T + alpha * (term1 + term2)

        forecasts: List[float] = [float(np.exp(f1))]
        prev = f1
        for _ in range(1, horizon):
            prev = omega + beta * prev
            forecasts.append(float(np.exp(prev)))

        return np.asarray(forecasts, dtype=float)
