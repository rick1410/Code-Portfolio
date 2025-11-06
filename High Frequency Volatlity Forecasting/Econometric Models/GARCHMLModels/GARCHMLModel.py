from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize  # kept for parity with original imports
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
import scipy


class GARCHMLNormalModel:
    """
    GARCHML with asymmetric/tanh effect and variance-in-mean under Normal errors.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of returns.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Innovation distribution ("Normal").
    returns : np.ndarray
        Stored return series.
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters if `optimize` succeeds.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (sum over observations).
    aic : Optional[float]
        Akaike Information Criterion at optimum.
    bic : Optional[float]
        Bayesian Information Criterion at optimum.
    convergence : bool
        Optimizer success flag.
    standard_errors : Optional[np.ndarray]
        Approximate standard errors from inverse Hessian (when computed).
    """

    model_name: str = "GARCHML"
    distribution: str = "Normal"

    returns: np.ndarray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: bool
    standard_errors: Optional[np.ndarray]

    def __init__(self, returns: np.ndarray) -> None:
        self.returns = np.asarray(returns, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def log_likelihood(self, params: Sequence[float]) -> float:
        """
        Average negative log-likelihood under Gaussian errors with variance-in-mean.

        Parameters
        ----------
        params : sequence of float
            (mu, omega, alpha, beta, delta, gamma, lamb).

        Returns
        -------
        float
            Average negative penalized log-likelihood (for minimization).
        """
        mu, omega, alpha, beta, delta, gamma, lamb = params
        T = len(self.returns)
        sigma2 = np.empty(T, dtype=float)
        sigma2[0] = float(np.var(self.returns[:50])) if T > 50 else float(np.var(self.returns))
        for t in range(1, T):
            term1 = alpha + delta * np.tanh(-gamma * self.returns[t - 1])
            z = (self.returns[t - 1] - mu - lamb * sigma2[t - 1]) / np.sqrt(sigma2[t - 1])
            sigma2[t] = omega + term1 * (z * z) + beta * sigma2[t - 1]
            if sigma2[t] <= 0.0:
                return 1e6
        ll = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + ((self.returns - mu - lamb * sigma2) ** 2) / sigma2)
        penalized_ll = ll - 0.001 * (gamma * gamma)
        return float(-np.mean(penalized_ll))

    @staticmethod
    def compute_aic_bic(total_ll: float, num_params: int, n_obs: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC.

        Parameters
        ----------
        total_ll : float
            Sum of log-likelihood contributions.
        num_params : int
            Number of estimated parameters.
        n_obs : int
            Sample size.

        Returns
        -------
        (float, float)
            (AIC, BIC).
        """
        aic = 2.0 * num_params - 2.0 * total_ll
        bic = np.log(n_obs) * num_params - 2.0 * total_ll
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, np.ndarray]:
        """
        Fit parameters via SLSQP under constraints.

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum if available else sensible defaults:
            {'mu': 0, 'omega': var(returns[:50])/50, 'alpha': 0.05, 'beta': 0.9,
             'delta': 0.01, 'gamma': 0.01, 'lamb': 0}.
        compute_metrics : bool, default False
            If True, compute total log-likelihood, AIC/BIC, and standard errors.

        Returns
        -------
        dict or tuple
            If `compute_metrics` and convergence:
            (params, AIC, BIC, loglik, SEs); else params dict.
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                base_var = float(np.var(self.returns[:50])) if len(self.returns) > 50 else float(np.var(self.returns))
                initial_params = {"mu": 0.0, "omega": base_var / 50.0 if base_var > 0.0 else 1e-6, "alpha": 0.05, "beta": 0.9, "delta": 0.01, "gamma": 0.01, "lamb": 0.0}
        keys = list(initial_params.keys()); x0 = list(initial_params.values())

        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[2] - abs(x[4])},  # alpha ≥ |delta|
            {'type': 'ineq', 'fun': lambda x: x[3]},              # beta ≥ 0
            {'type': 'ineq', 'fun': lambda x: x[5]},              # gamma ≥ 0
        ]

        res = scipy.optimize.minimize(self.log_likelihood, x0, method='SLSQP', constraints=constraints)
        self.convergence = bool(res.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))
        else:
            print(f"Warning: Optimization failed for {self.model_name}.")

        if compute_metrics and self.convergence:
            n_obs = int(len(self.returns))
            total_ll = float(-res.fun * n_obs)
            self.log_likelihood_value = total_ll
            k = len(x0)
            self.aic, self.bic = self.compute_aic_bic(total_ll, k, n_obs)
            H = approx_hess1(np.asarray(res.x, dtype=float), self.log_likelihood, args=())
            cov = inv(H) / n_obs
            ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            if not np.all(np.isfinite(ses)):
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = scipy.optimize.approx_fprime(np.asarray(res.x, dtype=float), self.log_likelihood, eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.maximum(np.diag(cov_alt), 0.0))
            self.standard_errors = ses.astype(float)
            return (self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors)
        return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int, num_simulations: int = 1000) -> np.ndarray:
        """
        Monte Carlo multi-step variance forecasts under fitted dynamics.

        Parameters
        ----------
        horizon : int
            Forecast horizon (≥ 1).
        num_simulations : int, default 1000
            Number of simulated paths for MC averaging.

        Returns
        -------
        np.ndarray
            Mean conditional variance path of length `horizon`.
        """
        if self.optimal_params is None:
            raise RuntimeError("Model must be optimized before forecasting.")
        np.random.seed(42)

        mu = float(self.optimal_params["mu"]); omega = float(self.optimal_params["omega"]); alpha = float(self.optimal_params["alpha"])
        beta = float(self.optimal_params["beta"]); delta = float(self.optimal_params["delta"]); gamma = float(self.optimal_params["gamma"]); lamb = float(self.optimal_params["lamb"])

        T = len(self.returns)
        sigma2 = np.empty(T, dtype=float)
        sigma2[0] = float(np.var(self.returns[:50])) if T > 50 else float(np.var(self.returns))
        for t in range(1, T):
            prev_r = self.returns[t - 1]
            prev_s2 = sigma2[t - 1]
            z = (prev_r - mu - lamb * prev_s2) / np.sqrt(prev_s2)
            term1 = alpha + delta * np.tanh(-gamma * prev_r)
            sigma2[t] = omega + term1 * (z * z) + beta * prev_s2
            if sigma2[t] <= 0.0:
                return np.full(horizon, np.nan, dtype=float)

        r_T = float(self.returns[-1])
        s2_T = float(sigma2[-1])
        zT = (r_T - mu - lamb * s2_T) / np.sqrt(s2_T)
        term1 = alpha + delta * np.tanh(-gamma * r_T)
        s2_1 = omega + term1 * (zT * zT) + beta * s2_T
        if s2_1 <= 0.0:
            return np.full(horizon, np.nan, dtype=float)

        r1 = mu + lamb * s2_1 + np.sqrt(s2_1) * zT

        sigma_paths = np.zeros((int(num_simulations), int(horizon)), dtype=float)
        for i in range(int(num_simulations)):
            prev_r = r1
            prev_s2 = s2_1
            for t in range(int(horizon)):
                z = np.random.randn()
                term1 = alpha + delta * np.tanh(-gamma * prev_r)
                new_s2 = omega + term1 * (z * z) + beta * prev_s2
                if new_s2 <= 0.0 or not np.isfinite(new_s2):
                    sigma_paths[i, t:] = np.nan
                    break
                sigma_paths[i, t] = new_s2
                prev_r = mu + lamb * new_s2 + np.sqrt(new_s2) * z
                prev_s2 = new_s2

        forecasted_sigma2 = np.nanmean(sigma_paths, axis=0)
        if np.any(np.isnan(forecasted_sigma2)):
            print(f"[NaN Warning] Detected invalid volatility path for horizon {horizon} in GARCHML")
        return forecasted_sigma2
