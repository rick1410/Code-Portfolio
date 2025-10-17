from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize  # kept for parity with original import style
from scipy.linalg import inv
import scipy.stats  # kept (norm/t) parity
from statsmodels.tools.numdiff import approx_hess1
from scipy.special import gammaln as lgamma
import scipy
from scipy.stats import t  # kept parity


class GARCHMLStudenttModel:
    """
    GARCHML with variance-in-mean, tanh asymmetry, and Student-t innovations.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of returns.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Innovation distribution ("Student t").
    returns : np.ndarray
        Stored return series.
    optimal_params : Optional[Dict[str, float]]
        Fitted parameters if `optimize` succeeds. Uses key 'nu1' for ν (compatibility).
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

    model_name: str = "GARCHML-t"
    distribution: str = "Student t"

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
        Average negative penalized log-likelihood for GARCH-M-L with Student-t errors.

        Parameters
        ----------
        params : sequence of float
            (mu, omega, alpha, beta, delta, gamma, lamb, nu).

        Returns
        -------
        float
            Average negative penalized log-likelihood (for minimization).
        """
        mu, omega, alpha, beta, delta, gamma, lamb, nu = params
        T = len(self.returns)
        sigma2 = np.empty(T, dtype=float)
        sigma2[0] = float(np.var(self.returns[:50])) if T > 50 else float(np.var(self.returns))
        for t_idx in range(1, T):
            term1 = alpha + delta * np.tanh(-gamma * self.returns[t_idx - 1])
            z = (self.returns[t_idx - 1] - mu - lamb * sigma2[t_idx - 1]) / np.sqrt(sigma2[t_idx - 1])
            sigma2[t_idx] = omega + term1 * (z * z) + beta * sigma2[t_idx - 1]
            if sigma2[t_idx] <= 0.0:
                return 1e6
        resid = (self.returns - mu - lamb * sigma2) / np.sqrt(sigma2)
        T2 = nu - 2.0
        ll = (lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * np.log(nu * np.pi * sigma2) - ((nu + 1.0) / 2.0) * np.log1p((resid * resid) / T2))
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
        Fit parameters via SLSQP under constraints (α ≥ |δ|, β ≥ 0, γ ≥ 0, ν > 2).

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum if available else defaults:
            {'mu': 0, 'omega': var(returns[:50])/50, 'alpha': 0.05, 'beta': 0.9,
             'delta': 0.01, 'gamma': 0.01, 'lamb': 0, 'nu1': 10}.
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
                omega0 = base_var / 50.0 if base_var > 0.0 else 1e-6
                initial_params = {'mu': 0.0, 'omega': omega0, 'alpha': 0.05, 'beta': 0.9, 'delta': 0.01, 'gamma': 0.01, 'lamb': 0.0, 'nu1': 10.0}
        keys = list(initial_params.keys()); x0 = list(initial_params.values())
        n_obs = int(len(self.returns))

        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[2] - abs(x[4])},      # alpha ≥ |delta|
            {'type': 'ineq', 'fun': lambda x: x[3]},                   # beta ≥ 0
            {'type': 'ineq', 'fun': lambda x: x[5]},                   # gamma ≥ 0
            {'type': 'ineq', 'fun': lambda x: x[7] - 2.000001},        # nu > 2
        ]

        res = scipy.optimize.minimize(self.log_likelihood, x0, method='SLSQP', constraints=constraints)
        self.convergence = bool(res.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))
        else:
            print(f"Warning: Optimization failed for {self.model_name}.")
        if compute_metrics and self.convergence:
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

    def simulate_returns(self, start_volatility: float, num_simulations: int, horizon: int) -> np.ndarray:
        """
        Simulate returns/variances and return mean variance path.

        Parameters
        ----------
        start_volatility : float
            Starting conditional variance σ²_T (> 0).
        num_simulations : int
            Number of Monte Carlo paths.
        horizon : int
            Number of steps ahead.

        Returns
        -------
        np.ndarray
            Mean of simulated variance paths (length `horizon`).
        """
        mu, omega, alpha, beta, delta, gamma, lamb, nu = [self.optimal_params[k] for k in ["mu", "omega", "alpha", "beta", "delta", "gamma", "lamb", "nu1"]]
        eps_floor = 1e-8
        sigma2 = max(float(start_volatility), eps_floor)

        simulated_returns = np.zeros((int(num_simulations), int(horizon)), dtype=float)
        sigma_2 = np.zeros((int(num_simulations), int(horizon)), dtype=float)

        for i in range(int(num_simulations)):
            local_sigma2 = sigma2
            for t_idx in range(int(horizon)):
                shock = np.random.standard_t(float(nu))
                local_sigma2 = max(local_sigma2, eps_floor)
                if not np.isfinite(local_sigma2):
                    local_sigma2 = 1e-4
                x_t = float(mu) + np.sqrt(local_sigma2) * shock
                simulated_returns[i, t_idx] = x_t
                term1 = float(alpha) + float(delta) * np.tanh(-float(gamma) * x_t)
                denom = np.sqrt(local_sigma2)
                if denom <= eps_floor or not np.isfinite(denom):
                    term2 = 0.0
                else:
                    term2 = ((x_t - float(mu) - float(lamb) * local_sigma2) / denom) ** 2
                next_sigma2 = float(omega) + term1 * term2 + float(beta) * local_sigma2
                if not np.isfinite(next_sigma2) or next_sigma2 <= 0.0:
                    next_sigma2 = 1e-6
                local_sigma2 = next_sigma2
                sigma_2[i, t_idx] = local_sigma2

        return np.mean(sigma_2, axis=0)

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Monte Carlo multi-step variance forecasts (Student-t errors).

        Parameters
        ----------
        horizon : int
            Forecast horizon (≥ 1).

        Returns
        -------
        np.ndarray
            Mean conditional variance path of length `horizon`.
        """
        num_simulations = 1000
        if self.optimal_params is None:
            raise RuntimeError("Optimize the model before forecasting.")
        mu, omega, alpha, beta, delta, gamma, lamb, nu = [self.optimal_params[k] for k in ["mu", "omega", "alpha", "beta", "delta", "gamma", "lamb", "nu1"]]
        returns = self.returns
        T = len(returns)

        sigma2 = np.zeros(T, dtype=float)
        sigma2[0] = float(np.var(returns[:50])) if T > 50 else float(np.var(returns))
        for t_idx in range(1, T):
            term1 = float(alpha) + float(delta) * np.tanh(-float(gamma) * returns[t_idx - 1])
            z = (returns[t_idx - 1] - float(mu) - float(lamb) * sigma2[t_idx - 1]) / np.sqrt(sigma2[t_idx - 1])
            sigma2[t_idx] = float(omega) + term1 * (z * z) + float(beta) * sigma2[t_idx - 1]
        start_sigma2 = float(sigma2[-1])

        forecasts = self.simulate_returns(start_sigma2, int(num_simulations), int(horizon))
        return forecasts
