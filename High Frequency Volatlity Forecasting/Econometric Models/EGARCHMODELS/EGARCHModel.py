from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1  # kept for API parity, not used here
from scipy.stats import norm

class EGARCHModel:
    """
    EGARCH(1,1) model with Normal innovations, estimated by MLE.

 
    Parameters
    ----------
    log_returns : ArrayLike
        1-D series of returns.

    Attributes
    ----------
    model_name : str
        Short model name ("EGARCH").
    distribution : str
        Innovation distribution ("Normal").
    log_returns : NDArray[np.float64]
        Stored input series as float array.
    optimal_params : Optional[Dict[str, float]]
        Optimized parameters {"omega","alpha","beta","gamma"}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at the optimum.
    aic : Optional[float]
        Akaike Information Criterion at the optimum.
    bic : Optional[float]
        Bayesian Information Criterion at the optimum.
    convergence : bool
        Optimizer success flag.
    standard_errors : Optional[NDArray[np.float64]]
        Approximate standard errors (outer-product-of-gradient fallback).
    """

    model_name: str = "EGARCH"
    distribution: str = "Normal"

    log_returns: NDArray[np.float64]
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: bool
    standard_errors: Optional[NDArray[np.float64]]

    def __init__(self, log_returns: ArrayLike) -> None:
        """Initialize the EGARCH(1,1) model with a returns series."""
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def log_likelihood(self, params: ArrayLike) -> float:
        """
        Negative average log-likelihood under Normality.

        Parameters
        ----------
        params : array-like of shape (4,)
            Parameter vector [omega, alpha, beta, gamma].

        Returns
        -------
        float
            Negative average log-likelihood (lower is better).
        """
        omega, alpha, beta, gamma = params
        r = self.log_returns
        T = len(r)

        log_sig2 = np.zeros(T, dtype=float)
        init_var = max(np.var(r[:50]), 1e-8)
        log_sig2[0] = float(np.log(init_var))

        for t in range(1, T):
            prev_var = float(np.exp(log_sig2[t - 1]))
            resid = r[t - 1] / np.sqrt(max(prev_var, 1e-8))
            log_sig2[t] = omega + alpha * (abs(resid) - np.sqrt(2 / np.pi)) + gamma * resid + beta * log_sig2[t - 1]

        sig2 = np.exp(log_sig2)
        ll = norm.logpdf(r, loc=0.0, scale=np.sqrt(sig2))
        return float(-np.mean(ll))

    def compute_aic_bic(self, total_ll: float, num_params: int) -> Tuple[float, float]:
        """
        Compute and return (AIC, BIC) from total log-likelihood.

        Parameters
        ----------
        total_ll : float
            Total (summed) log-likelihood at the optimum.
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        (float, float)
            (AIC, BIC).
        """
        n = len(self.log_returns)
        aic = 2 * num_params - 2 * total_ll
        bic = np.log(n) * num_params - 2 * total_ll
        return aic, bic

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, NDArray[np.float64]]:
        """
        Fit parameters via SLSQP with bounds and |beta|<1 constraint; optionally compute metrics and SEs.

        Parameters
        ----------
        initial_params : dict or None
            Initial guess {"omega","alpha","beta","gamma"}. If None, uses previous optimum if available,
            else defaults to {"omega":0.01,"alpha":0.1,"beta":0.7,"gamma":0.1}.
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
                initial_params = {"omega": 0.01, "alpha": 0.1, "beta": 0.7, "gamma": 0.1}

        keys = list(initial_params.keys())
        x0 = np.array(list(initial_params.values()), dtype=float)

        bounds = [(-100.0, 100.0), (0.0, 10.0), (None, None), (-10.0, 10.0)]
        cons = {'type': 'ineq', 'fun': lambda x: 1 - abs(x[2])}

        res = scipy.optimize.minimize(self.log_likelihood, x0, method='SLSQP', bounds=bounds, constraints=cons)
        self.convergence = bool(res.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, res.x))

        if compute_metrics and self.convergence:
            n = len(self.log_returns)
            total_ll = -res.fun * n
            self.log_likelihood_value = float(total_ll)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, len(x0))

            # Gradient at optimum, then OPG-style covariance (behavior preserved)
            g = scipy.optimize.approx_fprime(res.x, self.log_likelihood, np.sqrt(np.finfo(float).eps))
            try:
                cov = inv(np.outer(g, g)) / n
                self.standard_errors = np.sqrt(np.diag(cov))
            except Exception:
                self.standard_errors = np.full(len(x0), np.nan, dtype=float)

            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)

        return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> NDArray[np.float64]:
        """
        Monte Carlo multi-step-ahead variance forecasts using the fitted EGARCH recursion.

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

        np.random.seed(42)
        num_simulations = 1000
        tol = 1e-8

        omega = self.optimal_params["omega"]
        alpha = self.optimal_params["alpha"]
        beta = self.optimal_params["beta"]
        gamma = self.optimal_params["gamma"]

        r = self.log_returns
        T = len(r)
        log_sig2 = np.zeros(T, dtype=float)
        log_sig2[0] = float(np.log(max(np.var(r[:50]), tol)))

        for t in range(1, T):
            prev_var = float(np.exp(log_sig2[t - 1]))
            resid = r[t - 1] / np.sqrt(max(prev_var, tol))
            log_sig2[t] = omega + alpha * (abs(resid) - np.sqrt(2 / np.pi)) + gamma * resid + beta * log_sig2[t - 1]

        sig2_T = float(np.exp(log_sig2[-1]))
        r_T = float(r[-1])
        z_T = r_T / np.sqrt(sig2_T)

        log_next = omega + alpha * (abs(z_T) - np.sqrt(2 / np.pi)) + gamma * z_T + beta * np.log(max(sig2_T, tol))
        s2_1 = float(np.exp(min(max(log_next, np.log(tol)), 700)))

        if (not np.isfinite(s2_1)) or (s2_1 <= 0):
            return np.full(horizon, np.nan, dtype=float)

        forecasts = np.empty(horizon, dtype=float)
        forecasts[0] = s2_1
        if horizon == 1:
            return forecasts

        residuals = r / np.sqrt(np.exp(log_sig2))
        sim_paths = np.zeros((num_simulations, horizon - 1), dtype=float)

        for i in range(num_simulations):
            lv = float(np.log(s2_1))
            for j in range(horizon - 1):
                z = float(np.random.choice(residuals))
                lv = omega + alpha * (abs(z) - np.sqrt(2 / np.pi)) + gamma * z + beta * lv
                if not np.isfinite(lv):
                    sim_paths[i, j:] = np.nan
                    break
                sim_var = float(np.exp(min(max(lv, np.log(tol)), 700)))
                sim_paths[i, j] = sim_var

        forecasts[1:] = np.nanmean(sim_paths, axis=0)
        return forecasts
