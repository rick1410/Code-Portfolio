from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy
from scipy.stats import t as studentt
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1


class GAStModel:
    """
    GAS(1,1) volatility model with Student's t innovations.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of returns.

    Attributes
    ----------
    model_name : str
        Human-readable model name.
    distribution : str
        Innovation distribution ('Student t').
    log_returns : np.ndarray
        Return series (float64).
    optimal_params : Optional[Dict[str, float]]
        Estimated parameters with keys {'omega','alpha','beta','nu1'}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum.
    aic : Optional[float]
        Akaike Information Criterion.
    bic : Optional[float]
        Bayesian Information Criterion.
    convergence : Optional[bool]
        Optimizer success flag.
    standard_errors : Optional[np.ndarray]
        Approximate standard errors from inverse Hessian (if computed).
    """

    model_name: str = "GAS-t"
    distribution: str = "Student t"

    log_returns: np.ndarray
    optimal_params: Optional[Dict[str, float]]
    log_likelihood_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    convergence: Optional[bool]
    standard_errors: Optional[np.ndarray]

    def __init__(self, log_returns: np.ndarray) -> None:
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = None
        self.standard_errors = None

    def log_likelihood(self, params: Sequence[float]) -> float:
        """
        Average negative log-likelihood for GAS(1,1) with Student's t errors.

        Parameters
        ----------
        params : sequence of float
            (omega, alpha, beta, nu).

        Returns
        -------
        float
            Average negative log-likelihood (to minimize).
        """
        omega, alpha, beta, nu = params
        r = self.log_returns
        T = len(r)

        f = np.zeros(T, dtype=float)
        f[0] = np.log(np.var(r[:50]) if T > 50 else np.var(r))
        for t in range(1, T):
            r_tm1 = r[t - 1]
            f_tm1 = f[t - 1]
            term1 = -0.5
            term2 = (nu + 1.0) / 2.0
            term3 = (r_tm1 ** 2) / ((nu - 2.0) * np.exp(f_tm1) + r_tm1 ** 2)
            nabla_t = term1 + term2 * term3
            f[t] = omega + beta * f_tm1 + alpha * nabla_t

        l = studentt.logpdf(r, df=nu, scale=np.sqrt(np.exp(f)))
        return float(-np.mean(l))

    def compute_aic_bic(self, log_likelihood: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC (preserving original formulation).

        Parameters
        ----------
        log_likelihood : float
            Mean log-likelihood (as used in original code path).
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        (float, float)
            (AIC, BIC).
        """
        n = len(self.log_returns)
        aic = 2.0 * num_params - 2.0 * (log_likelihood * n)
        bic = np.log(n) * num_params - 2.0 * log_likelihood
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = True) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, np.ndarray]:
        """
        Estimate parameters via SLSQP subject to |beta|<1 and nu>2.

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum else defaults
            {'omega': var(r[:50])*(1-0.1-0.7), 'alpha':0.1, 'beta':0.7, 'nu1':3}.
        compute_metrics : bool, default True
            If True, also compute AIC/BIC, total log-likelihood and standard errors.

        Returns
        -------
        dict or tuple
            Params dict; or (params, AIC, BIC, total_loglik, SEs) when `compute_metrics=True`.
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                v0 = float(np.var(self.log_returns[:50]) if len(self.log_returns) > 50 else np.var(self.log_returns))
                initial_params = {"omega": v0 * (1.0 - 0.1 - 0.7), "alpha": 0.1, "beta": 0.7, "nu1": 3.0}

        keys: List[str] = list(initial_params.keys())
        x0: List[float] = list(initial_params.values())

        bounds = [(-100.0, 100.0), (0.0, 10.0), (0.0, 10.0), (2.000001, 100.0)]
        constraints = {'type': 'ineq', 'fun': lambda x: 1.0 - abs(x[2])}

        result = scipy.optimize.minimize(self.log_likelihood, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in result.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")

        if compute_metrics and self.convergence:
            self.log_likelihood_value = float(-result.fun * len(self.log_returns))
            num_params = len(x0)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, num_params)

            hessian = approx_hess1(list(self.optimal_params.values()), self.log_likelihood, args=())
            hessian_inv = inv(hessian) / max(len(self.log_returns), 1)
            self.standard_errors = np.sqrt(np.maximum(np.diag(hessian_inv), 0.0))
            if np.isnan(self.standard_errors).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                grad = scipy.optimize.approx_fprime(list(self.optimal_params.values()), self.log_likelihood, eps)
                hessian_inv_alt = np.outer(grad, grad)
                self.standard_errors = np.sqrt(np.maximum(np.diag(hessian_inv_alt), 0.0))

            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors
        return self.optimal_params  # type: ignore[return-value]

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Multi-step variance forecast via residual bootstrap:
          1) Exact one-step score update,
          2) For steps ≥ 2, resample standardized residuals and update.

        Parameters
        ----------
        horizon : int
            Number of steps ahead.

        Returns
        -------
        np.ndarray
            Forecasted variances of length `horizon`.
        """
        if self.optimal_params is None: raise RuntimeError("Model must be optimized before forecasting.")

        omega, alpha, beta, nu = [self.optimal_params[k] for k in ["omega", "alpha", "beta", "nu1"]]
        r = self.log_returns
        T = len(r)

        f = np.zeros(T, dtype=float)
        f[0] = np.log(np.var(r[:50]) if T > 50 else np.var(r))
        for t in range(1, T):
            rt = r[t - 1]
            f_tm1 = f[t - 1]
            term1 = -0.5
            term2 = (nu + 1.0) / 2.0
            term3 = (rt**2) / ((nu - 2.0) * np.exp(f_tm1) + rt**2)
            nabla = term1 + term2 * term3
            f[t] = omega + beta * f_tm1 + alpha * nabla

        z = r / np.sqrt(np.exp(f))

        rT = r[-1]
        fT = f[-1]
        term3_T = (rT**2) / ((nu - 2.0) * np.exp(fT) + rT**2)
        nabla_T = -0.5 + (nu + 1.0) / 2.0 * term3_T
        f1 = omega + beta * fT + alpha * nabla_T
        forecasts: List[float] = [float(np.exp(f1))]
        f_prev = f1

        for _ in range(1, int(horizon)):
            zt = float(np.random.choice(z))
            eps = zt * np.sqrt(float(np.exp(f_prev)))
            term3 = (eps**2) / ((nu - 2.0) * np.exp(f_prev) + eps**2)
            nabla = -0.5 + (nu + 1.0) / 2.0 * term3
            f_next = omega + beta * f_prev + alpha * nabla
            forecasts.append(float(np.exp(f_next)))
            f_prev = f_next

        return np.array(forecasts, dtype=float)
