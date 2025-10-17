from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.optimize
from scipy.stats import t as studentt
from scipy.special import gamma
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1

class EGARCHtModel:
    """
    EGARCH(1,1) model with Student-t innovations, estimated by MLE.

    Model
    -----
    r_t = ε_t,  ε_t | 𝔽_{t-1} ~ t_ν(0, σ_t^2)
    log(σ_t^2) = ω + α (|z_{t-1}| - E|T_ν|) + γ z_{t-1} + β log(σ_{t-1}^2),  z_t = r_t / σ_t

    Parameters
    ----------
    log_returns : ArrayLike
        1-D series of returns.

    Attributes
    ----------
    model_name : str
        Short model name ("EGARCH-t").
    distribution : str
        Innovation distribution ("Student t").
    log_returns : NDArray[np.float64]
        Stored input series as float array.
    optimal_params : Optional[Dict[str, float]]
        Optimized parameters {"omega","alpha","beta","gamma_p","nu1"} (nu1 ≡ ν).
    log_likelihood_value : Optional[float]
        Total log-likelihood at the optimum.
    aic : Optional[float]
        Akaike Information Criterion at the optimum.
    bic : Optional[float]
        Bayesian Information Criterion at the optimum.
    convergence : bool
        Optimizer success flag.
    standard_errors : Optional[NDArray[np.float64]]
        Approximate standard errors (inverse Hessian with fallback).
    """

    model_name: str = "EGARCH-t"
    distribution: str = "Student t"

    log_returns: NDArray[np.float64]
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: bool
    standard_errors: Optional[NDArray[np.float64]]

    def __init__(self, log_returns: ArrayLike) -> None:
        """Initialize the EGARCH(1,1)-t model with a returns series."""
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def log_likelihood(self, params: ArrayLike, tol: float = 1e-8) -> float:
        """
        Negative average log-likelihood under Student-t innovations.

        Parameters
        ----------
        params : array-like of shape (5,)
            Parameter vector [omega, alpha, beta, gamma_p, nu].
        tol : float, default 1e-8
            Numerical floor for variances.

        Returns
        -------
        float
            Negative average log-likelihood (lower is better).
        """
        omega, alpha, beta, gamma_p, nu = params
        r = self.log_returns
        T = len(r)

        log_sig2 = np.zeros(T, dtype=float)
        log_sig2[0] = float(np.log(max(np.var(r[:50]), tol)))

        E_abs_t = (np.sqrt(nu) * gamma((nu + 1) / 2)) / (np.sqrt(np.pi) * gamma(nu / 2))

        for t in range(1, T):
            prev_var = float(np.exp(log_sig2[t - 1]))
            resid = r[t - 1] / np.sqrt(max(prev_var, tol))
            log_sig2[t] = omega + alpha * (abs(resid) - E_abs_t) + gamma_p * resid + beta * log_sig2[t - 1]

        sig2 = np.maximum(np.exp(log_sig2), tol)
        ll = studentt.logpdf(r, df=nu, loc=0.0, scale=np.sqrt(sig2))
        return float(-np.mean(ll))

    def compute_aic_bic(self, total_ll: float, k: int, n_obs: int) -> Tuple[float, float]:
        """
        Compute and return (AIC, BIC) from total log-likelihood.

        Parameters
        ----------
        total_ll : float
            Total (summed) log-likelihood at the optimum.
        k : int
            Number of estimated parameters.
        n_obs : int
            Number of observations used in estimation.

        Returns
        -------
        (float, float)
            (AIC, BIC).
        """
        aic = 2 * k - 2 * total_ll
        bic = np.log(n_obs) * k - 2 * total_ll
        return aic, bic

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, NDArray[np.float64]]:
        """
        Fit parameters via SLSQP with bounds and |beta|<1 constraint; optionally compute metrics and SEs.

        Parameters
        ----------
        initial_params : dict or None
            Initial guess {"omega","alpha","beta","gamma_p","nu1"}. If None, uses previous optimum if available,
            else defaults to {"omega":var(r[:50])*0.1,"alpha":0.1,"beta":0.7,"gamma_p":0.1,"nu1":5.0}.
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
            initial_params = getattr(self, 'optimal_params', None) or {"omega": float(np.var(self.log_returns[:50]) * 0.1), "alpha": 0.1, "beta": 0.7, "gamma_p": 0.1, "nu1": 5.0}

        keys = list(initial_params.keys())
        x0 = np.array(list(initial_params.values()), dtype=float)

        bounds = [(-100.0, 100.0), (0.0, 10.0), (0.0, 10.0), (-10.0, 10.0), (2.0001, None)]
        cons = [{'type': 'ineq', 'fun': lambda x: 1 - abs(x[2])}]

        res = scipy.optimize.minimize(fun=self.log_likelihood, x0=x0, method='SLSQP', bounds=bounds, constraints=cons)
        self.convergence = bool(res.success)
        if not self.convergence:
            print(f"Warning: {self.model_name} failed: {res.message}")
            return self.optimal_params

        self.optimal_params = dict(zip(keys, res.x))

        if compute_metrics:
            n_obs = len(self.log_returns)
            total_ll = -res.fun * n_obs
            self.log_likelihood_value = float(total_ll)
            k = len(x0)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, k, n_obs)

            H = approx_hess1(res.x, self.log_likelihood, args=())
            cov = inv(H) / n_obs
            ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            if np.isnan(ses).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = scipy.optimize.approx_fprime(res.x, self.log_likelihood, eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.maximum(np.diag(cov_alt), 0.0))
            self.standard_errors = ses.astype(float)

            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)

        return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> NDArray[np.float64]:
        """
        Monte Carlo multi-step-ahead variance forecasts using the fitted EGARCH-t recursion.

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
            raise RuntimeError("Optimize before forecasting.")

        np.random.seed(42)
        tol = 1e-8
        max_log = 700.0
        num_simulations = 100

        omega = self.optimal_params["omega"]
        alpha = self.optimal_params["alpha"]
        beta = self.optimal_params["beta"]
        gamma_p = self.optimal_params["gamma_p"]
        nu = self.optimal_params["nu1"]

        r = self.log_returns
        T = len(r)
        log_sig2 = np.zeros(T, dtype=float)
        log_sig2[0] = float(np.log(max(np.var(r[:50]), tol)))
        E_abs_t = (np.sqrt(nu) * gamma((nu + 1) / 2)) / (np.sqrt(np.pi) * gamma(nu / 2))

        for t in range(1, T):
            prev_var = float(np.exp(log_sig2[t - 1]))
            resid = r[t - 1] / np.sqrt(max(prev_var, tol))
            log_sig2[t] = omega + alpha * (abs(resid) - E_abs_t) + gamma_p * resid + beta * log_sig2[t - 1]

        sig2_T = float(np.exp(log_sig2[-1]))
        zT = float(r[-1] / np.sqrt(max(sig2_T, tol)))
        next_log = omega + alpha * (abs(zT) - E_abs_t) + gamma_p * zT + beta * log_sig2[-1]
        next_log = float(np.clip(next_log, np.log(tol), max_log))
        s2_1 = float(np.exp(next_log))

        forecasts = np.empty(horizon, dtype=float)
        forecasts[0] = s2_1
        if horizon == 1:
            return forecasts

        sims = np.zeros((num_simulations, horizon - 1), dtype=float)
        for i in range(num_simulations):
            lv = next_log
            for j in range(horizon - 1):
                z = float(studentt.rvs(df=nu))
                eps = z * np.sqrt(np.exp(min(max(lv, np.log(tol)), max_log)))
                lv = omega + alpha * (abs(eps) - E_abs_t) + gamma_p * z + beta * lv
                lv = float(np.clip(lv, np.log(tol), max_log))
                sims[i, j] = float(np.exp(lv))

        forecasts[1:] = sims.mean(axis=0)
        return forecasts
