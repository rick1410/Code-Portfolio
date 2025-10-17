from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma, digamma, polygamma
import scipy
from statsmodels.tools.numdiff import approx_hess1
from scipy.linalg import inv

class EGARCHastModel:
    """
    EGARCH(1,1) model with Asymmetric Student-t (AST) innovations, estimated by MLE.

    Model
    -----
    r_t = ε_t,  ε_t | 𝔽_{t-1} ~ AST(δ, ν1, ν2) scaled by σ_t
    log(σ_t^2) = ω + α (|r_{t-1}| - E|AST|) + γ r_{t-1} + β log(σ_{t-1}^2)

    Parameters
    ----------
    log_returns : ArrayLike
        1-D time series of returns.

    Attributes
    ----------
    model_name : str
        Short model name ("EGARCH-AST").
    distribution : str
        Innovation distribution ("AST").
    log_returns : NDArray[np.float64]
        Stored input series as float array.
    optimal_params : Optional[Dict[str, float]]
        Optimized parameters {"omega","alpha","beta","gamma","delta","v1","v2"}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at the optimum.
    aic : Optional[float]
        Akaike Information Criterion at the optimum (per original formula).
    bic : Optional[float]
        Bayesian Information Criterion at the optimum (per original formula).
    convergence : Optional[bool]
        Optimizer success flag.
    standard_errors : Optional[NDArray[np.float64]]
        Approximate standard errors (inverse Hessian with fallback).
    """

    model_name: str = "EGARCH-AST"
    distribution: str = "AST"

    log_returns: NDArray[np.float64]
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: Optional[bool]
    standard_errors: Optional[NDArray[np.float64]]

    def __init__(self, log_returns: ArrayLike) -> None:
        """Initialize the EGARCH-AST model with a returns series."""
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = None
        self.standard_errors = None

    def K(self, v: float) -> float:
        """Return K(v) = Γ((v+1)/2) / (√(π v) Γ(v/2))."""
        return float(gamma((v + 1) / 2) / (np.sqrt(np.pi * v) * gamma(v / 2)))

    def B(self, delta: float, v1: float, v2: float) -> float:
        """Return B(δ, ν1, ν2) = δ K(ν1) + (1-δ) K(ν2)."""
        return float(delta * self.K(v1) + (1 - delta) * self.K(v2))

    def alpha_star(self, delta: float, v1: float, v2: float) -> float:
        """Return α* = δ K(ν1) / B(δ, ν1, ν2)."""
        B = self.B(delta, v1, v2)
        return float((delta * self.K(v1)) / B)

    def m(self, delta: float, v1: float, v2: float) -> float:
        """Return m(δ, ν1, ν2) used in AST location adjustment."""
        alpha_star = self.alpha_star(delta, v1, v2)
        B = self.B(delta, v1, v2)
        return float(4 * B * (-(alpha_star**2) * v1 / (v1 - 1) + (1 - alpha_star) ** 2 * v2 / (v2 - 1)))

    def s(self, delta: float, v1: float, v2: float) -> float:
        """Return s(δ, ν1, ν2) used in AST scale adjustment."""
        alpha_star = self.alpha_star(delta, v1, v2)
        m_val = self.m(delta, v1, v2)
        return float(np.sqrt(4 * (delta * alpha_star**2 * v1 / (v1 - 2) + (1 - delta) * (1 - alpha_star) ** 2 * v2 / (v2 - 2)) - m_val**2))

    def I_t(self, r_t: float, mu_t: float, h_t: float, m_val: float, s_val: float) -> int:
        """Indicator I_t = 1{ m + s (r_t - μ_t)/sqrt(h_t) > 0 }."""
        return 1 if m_val + s_val * (r_t - mu_t) / np.sqrt(h_t) > 0 else 0

    def expected_abs_ast(self, delta: float, v1: float, v2: float) -> float:
        """
        Expected absolute value E|AST| for the AST distribution.

        Parameters
        ----------
        delta : float
            Skewness weight (0 ≤ δ ≤ 1).
        v1 : float
            Left-tail degrees of freedom (> 1).
        v2 : float
            Right-tail degrees of freedom (> 1).

        Returns
        -------
        float
            E|AST|.
        """
        alpha_star = self.alpha_star(delta, v1, v2)
        E_T_v1 = np.sqrt(v1 / np.pi) * gamma(0.5) * gamma((v1 - 1) / 2) / gamma(v1 / 2)
        E_T_v2 = np.sqrt(v2 / np.pi) * gamma(0.5) * gamma((v2 - 1) / 2) / gamma(v2 / 2)
        return float(delta * 2 * alpha_star * E_T_v1 + (1 - delta) * 2 * (1 - alpha_star) * E_T_v2)

    def log_likelihood(self, params: ArrayLike) -> float:
        """
        Negative average log-likelihood under AST innovations.

        Parameters
        ----------
        params : array-like of shape (7,)
            Parameter vector [omega, alpha, beta, gamma, delta, v1, v2].

        Returns
        -------
        float
            Negative average log-likelihood (lower is better).
        """
        r = self.log_returns
        T = len(r)
        omega, alpha, beta, gamma_p, delta, v1, v2 = params

        log_sig2 = np.zeros(T, dtype=float)
        log_sig2[0] = float(np.log(np.var(r[:50]) if T > 50 else np.var(r)))

        for t in range(1, T):
            E_abs_ast = self.expected_abs_ast(delta, v1, v2)
            log_sig2[t] = omega + alpha * (abs(r[t - 1]) - E_abs_ast) + gamma_p * r[t - 1] + beta * log_sig2[t - 1]

        sig2 = np.exp(log_sig2)

        ll = np.zeros(T, dtype=float)
        for t in range(T):
            h_t = sig2[t]
            s_val = self.s(delta, v1, v2)
            m_val = self.m(delta, v1, v2)
            alpha_st = self.alpha_star(delta, v1, v2)
            B_val = self.B(delta, v1, v2)
            I_val = self.I_t(r[t], 0.0, h_t, m_val, s_val)
            term1 = (v1 + 1) / 2 * np.log(1 + (1 / v1) * ((m_val + s_val * (r[t] / np.sqrt(h_t)) / (2 * alpha_st)) ** 2))
            term2 = (v2 + 1) / 2 * np.log(1 + (1 / v2) * ((m_val + s_val * (r[t] / np.sqrt(h_t)) / (2 * (1 - alpha_st))) ** 2))
            ll[t] = np.log(s_val) + np.log(B_val) - 0.5 * np.log(h_t) - (1 - I_val) * term1 - I_val * term2

        return float(-np.mean(ll))

    def compute_aic_bic(self, log_likelihood: float, num_params: int) -> Tuple[float, float]:
        """
        Compute and return (AIC, BIC) following the original implementation.

        Parameters
        ----------
        log_likelihood : float
            Total log-likelihood at the optimum (as passed in by caller).
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        (float, float)
            (AIC, BIC).
        """
        aic = 2 * num_params - 2 * (log_likelihood * len(self.log_returns))
        bic = np.log(len(self.log_returns)) * num_params - 2 * log_likelihood
        return aic, bic

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, NDArray[np.float64]]:
        """
        Fit parameters via SLSQP with |beta|<1 constraint; optionally compute metrics and SEs.

        Parameters
        ----------
        initial_params : dict or None
            Initial guess {"omega","alpha","beta","gamma","delta","v1","v2"}. If None, uses previous optimum if available,
            else defaults to {"omega":0.134,"alpha":0.043,"beta":0.915,"gamma":-0.08,"delta":0.5,"v1":3.74,"v2":3.74}.
        compute_metrics : bool, default False
            If True, also computes total log-likelihood, AIC, BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If `compute_metrics` is False: dict with optimal parameters.
            If `compute_metrics` is True and converged:
                (optimal_params, aic, bic, log_likelihood_value, standard_errors).
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {"omega": 0.134, "alpha": 0.043, "beta": 0.915, "gamma": -0.08, "delta": 0.5, "v1": 3.74, "v2": 3.74}

        keys = list(initial_params.keys())
        x0 = np.array(list(initial_params.values()), dtype=float)

        bounds = [(-100.0, 100.0), (0.0, 10.0), (0.0, 10.0), (-10.0, 10.0), (0.0, 1.0), (0.0001, None), (0.0001, None)]
        constraints = {'type': 'ineq', 'fun': lambda x: 1 - abs(x[2])}

        result = scipy.optimize.minimize(self.log_likelihood, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, result.x))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")

        if compute_metrics and self.convergence:
            self.log_likelihood_value = float(-result.fun * len(self.log_returns))
            num_params = len(x0)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, num_params)

            hessian = approx_hess1(list(self.optimal_params.values()), self.log_likelihood, args=())
            hessian_inv = inv(hessian) / len(self.log_returns)
            ses = np.sqrt(np.maximum(np.diag(hessian_inv), 0.0))

            if np.isnan(ses).any():
                epsilon = float(np.sqrt(np.finfo(float).eps))
                grad = scipy.optimize.approx_fprime(list(self.optimal_params.values()), self.log_likelihood, epsilon)
                hessian_inv_alt = np.outer(grad, grad)
                ses = np.sqrt(np.maximum(np.diag(hessian_inv_alt), 0.0))

            self.standard_errors = ses.astype(float)
            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)
        else:
            return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> NDArray[np.float64]:
        """
        Multi-step variance forecast: step 1 uses the actual last return; steps ≥ 2 bootstrap standardized residuals.

        Parameters
        ----------
        horizon : int
            Number of steps ahead (h >= 1).

        Returns
        -------
        np.ndarray
            Array of length `horizon` with variance forecasts.

        Raises
        ------
        RuntimeError
            If the model has not been optimized yet.
        """
        if self.optimal_params is None:
            raise RuntimeError("Model must be optimized before forecasting.")

        omega = self.optimal_params["omega"]
        alpha = self.optimal_params["alpha"]
        beta = self.optimal_params["beta"]
        gamma_p = self.optimal_params["gamma"]
        delta = self.optimal_params["delta"]
        v1 = self.optimal_params["v1"]
        v2 = self.optimal_params["v2"]

        r = self.log_returns
        T = len(r)
        log_sig2 = np.zeros(T, dtype=float)
        log_sig2[0] = float(np.log(np.var(r[:50]) if T > 50 else np.var(r)))
        e_abs = self.expected_abs_ast(delta, v1, v2)
        for t in range(1, T):
            log_sig2[t] = omega + alpha * (abs(r[t - 1]) - e_abs) + gamma_p * r[t - 1] + beta * log_sig2[t - 1]

        sig = np.sqrt(np.exp(log_sig2))
        residuals = r / np.where(sig > 0, sig, np.finfo(float).eps)

        r_T = r[-1]
        last_log = log_sig2[-1]
        next_log = omega + alpha * (abs(r_T) - e_abs) + gamma_p * r_T + beta * last_log

        forecasts = [float(np.exp(next_log))]
        lv = float(next_log)

        for _ in range(1, horizon):
            z = float(np.random.choice(residuals))
            eps = z * np.sqrt(np.exp(lv))
            lv = omega + alpha * (abs(eps) - e_abs) + gamma_p * eps + beta * lv
            forecasts.append(float(np.exp(lv)))

        return np.array(forecasts, dtype=float)
