
from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.special import digamma, polygamma, gamma

class EGARCHegb2Model:
    """
    EGARCH(1,1) model with EGB2 innovations, estimated by MLE.

    Parameters
    ----------
    data : ArrayLike
        1-D time series of returns.

    Attributes
    ----------
    model_name : str
        Short model name ("EGARCH-EGB2").
    distribution : str
        Innovation distribution ("EGB2").
    data : NDArray[np.float64]
        Stored input series as float array.
    optimal_params : Optional[Dict[str, float]]
        Optimized parameters {"omega","alpha","beta","gamma","p","q"}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at the optimum.
    aic : Optional[float]
        Akaike Information Criterion at the optimum (per original implementation).
    bic : Optional[float]
        Bayesian Information Criterion at the optimum (per original implementation).
    convergence : Optional[bool]
        Optimizer success flag.
    standard_errors : Optional[NDArray[np.float64]]
        Approximate standard errors (inverse Hessian with fallback).
    """

    model_name: str = "EGARCH-EGB2"
    distribution: str = "EGB2"

    data: NDArray[np.float64]
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: Optional[bool]
    standard_errors: Optional[NDArray[np.float64]]

    def __init__(self, data: ArrayLike) -> None:
        """Initialize the EGARCH-EGB2 model with a returns series."""
        self.data = np.asarray(data, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = None
        self.standard_errors = None

    def log_likelihood(self, params: ArrayLike) -> float:
        """
        Negative average log-likelihood under EGB2 innovations.

        Parameters
        ----------
        params : array-like of shape (6,)
            Parameter vector [omega, alpha, beta, gamma, p, q].

        Returns
        -------
        float
            Negative average log-likelihood (lower is better).
        """
        omega, alpha, beta, gamma_p, p, q = params
        r = self.data
        T = len(r)
        mu_t = 0.0

        log_sig2 = np.zeros(T, dtype=float)
        init_var = np.var(r[:50]) if T > 50 else np.var(r)
        log_sig2[0] = float(np.log(max(init_var, 1e-12)))

        Delta = float(digamma(p) - digamma(q))
        Omega = float(polygamma(1, p) + polygamma(1, q))
        E_abs_egb2 = 0.0  # placeholder per original code

        for t in range(1, T):
            log_sig2[t] = omega + alpha * (abs(r[t - 1]) - E_abs_egb2) + gamma_p * r[t - 1] + beta * log_sig2[t - 1]

        sig2 = np.exp(log_sig2)

        ll = 0.0
        norm_const = gamma(p) * gamma(q) / gamma(p + q)
        root_Omega = np.sqrt(Omega)
        for t in range(T):
            rt = r[t]
            ht = sig2[t]
            z = root_Omega * (rt - mu_t) / np.sqrt(ht) + Delta
            ll += 0.5 * np.log(Omega) + p * z - 0.5 * np.log(ht) - np.log(norm_const) - (p + q) * np.log1p(np.exp(z))

        return float(-ll / T)

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
            (AIC, BIC) with original code's formulas (includes printing).
        """
        aic = 2 * num_params - 2 * (log_likelihood * len(self.data))
        bic = np.log(len(self.data)) * num_params - 2 * log_likelihood
        print(f"AIC: {aic}", f"BIC: {bic}")
        return aic, bic

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = True) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, NDArray[np.float64]]:
        """
        Fit parameters via SLSQP with |beta|<1 constraint; optionally compute metrics and SEs.

        Parameters
        ----------
        initial_params : dict or None
            Initial guess {"omega","alpha","beta","gamma","p","q"}. If None, uses previous optimum if available,
            else defaults to {"omega":var(data[:50])*(1-0.1-0.1-0.7), "alpha":0.1, "beta":0.7, "gamma":0.1, "p":4, "q":4}.
        compute_metrics : bool, default True
            If True, also computes total log-likelihood, AIC, BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If `compute_metrics` is False or not converged: dict with optimal parameters (may be previous).
            If `compute_metrics` is True and converged:
                (optimal_params, aic, bic, log_likelihood_value, standard_errors).
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                v0 = np.var(self.data[:50]) if self.data.size > 50 else np.var(self.data)
                initial_params = {"omega": float(v0 * (1 - 0.1 - 0.1 - 0.7)), "alpha": 0.1, "beta": 0.7, "gamma": 0.1, "p": 4.0, "q": 4.0}

        keys = list(initial_params.keys())
        x0 = np.array(list(initial_params.values()), dtype=float)

        bounds = ((0.0, 100.0), (0.0, 10.0), (0.0, 10.0), (-10.0, 10.0), (0.001, 100.0), (0.001, 100.0))
        constraints = ({'type': 'ineq', 'fun': lambda x: 1 - np.abs(x[2])},)

        result = scipy.optimize.minimize(self.log_likelihood, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, result.x))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")

        if compute_metrics and self.convergence:
            self.log_likelihood_value = float(-result.fun * len(self.data))
            num_params = len(x0)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, num_params)

            H = approx_hess1(list(self.optimal_params.values()), self.log_likelihood, args=())
            Hinv = inv(H) / len(self.data)
            ses = np.sqrt(np.maximum(np.diag(Hinv), 0.0))

            if np.isnan(ses).any():
                epsilon = float(np.sqrt(np.finfo(float).eps))
                grad = scipy.optimize.approx_fprime(list(self.optimal_params.values()), self.log_likelihood, epsilon)
                Hinv_alt = np.outer(grad, grad)
                ses = np.sqrt(np.maximum(np.diag(Hinv_alt), 0.0))

            self.standard_errors = ses.astype(float)
            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)
        else:
            return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> NDArray[np.float64]:
        """
        Multi-step variance forecasts via EGARCH recursion with bootstrap residuals.

        Step 1 uses the last observed return; steps ≥ 2 draw residuals from the standardized
        in-sample residuals to propagate the recursion.

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
        p = self.optimal_params["p"]
        q = self.optimal_params["q"]

        T = len(self.data)
        log_sig2 = np.zeros(T, dtype=float)
        log_sig2[0] = float(np.log(np.var(self.data[:50]) if T > 50 else np.var(self.data)))
        E_abs = 0.0  # placeholder, consistent with estimation path

        for t in range(1, T):
            log_sig2[t] = omega + alpha * (abs(self.data[t - 1]) - E_abs) + gamma_p * self.data[t - 1] + beta * log_sig2[t - 1]

        sig = np.sqrt(np.exp(log_sig2))
        residuals = self.data / np.where(sig > 0, sig, np.finfo(float).eps)

        rT = self.data[-1]
        last_log = log_sig2[-1]
        next_log = omega + alpha * (abs(rT) - E_abs) + gamma_p * rT + beta * last_log

        forecasts = [float(np.exp(next_log))]
        lv = float(next_log)

        for _ in range(1, horizon):
            z = float(np.random.choice(residuals))
            eps = z * np.sqrt(np.exp(lv))
            lv = omega + alpha * (abs(eps) - E_abs) + gamma_p * eps + beta * lv
            forecasts.append(float(np.exp(lv)))

        return np.array(forecasts, dtype=float)
