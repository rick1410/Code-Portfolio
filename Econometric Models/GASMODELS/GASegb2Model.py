from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy
from scipy.special import gamma, digamma, polygamma
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1


class GASegb2Model:
    """
    Generalized Autoregressive Score (GAS) model with EGB2 innovations.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of returns.

    Attributes
    ----------
    model_name : str
        Human-readable name of the model.
    distribution : str
        Innovation distribution ("EGB2").
    log_returns : np.ndarray
        Return series (float64).
    optimal_params : Optional[Dict[str, float]]
        Fitted parameter dictionary with keys {"omega","alpha","beta","p","q"}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (sum across observations).
    aic : Optional[float]
        Akaike Information Criterion.
    bic : Optional[float]
        Bayesian Information Criterion.
    convergence : Optional[bool]
        Optimizer success flag.
    standard_errors : Optional[np.ndarray]
        Approximate standard errors from inverse Hessian (if computed).
    """

    model_name: str = "GAS-EGB2"
    distribution: str = "EGB2"

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
        Average negative log-likelihood for GAS-EGB2 (to minimize).

        Parameters
        ----------
        params : sequence of float
            (omega, alpha, beta, p, q).

        Returns
        -------
        float
            Average negative log-likelihood.
        """
        omega, alpha, beta, p, q = params
        r = self.log_returns
        T = len(r)
        mu_t = 0.0

        f = np.zeros(T, dtype=float)
        f[0] = np.log(np.var(r[:50]) if T > 50 else np.var(r))

        Delta = float(digamma(p) - digamma(q))
        Omega = float(polygamma(1, p) + polygamma(1, q))

        for t in range(1, T):
            r_tm1 = r[t - 1]
            h_tm1 = float(np.exp(f[t - 1]))
            h_sqrt = np.sqrt(max(h_tm1, 1e-12))
            exp_arg = float(np.sqrt(Omega) * r_tm1 / h_sqrt + Delta)
            num = np.sqrt(Omega) * (-q - p) * r_tm1 * np.exp(exp_arg)
            den = 2.0 * h_sqrt * (np.exp(exp_arg) + 1.0)
            nabla_t = num / den - 0.5 - np.sqrt(Omega) * p * r_tm1 / (2.0 * h_sqrt)
            f[t] = omega + beta * f[t - 1] + alpha * nabla_t

        ll_sum = 0.0
        norm_const = float(gamma(p) * gamma(q) / gamma(p + q))
        for t in range(T):
            h_t = float(np.exp(f[t]))
            h_sqrt = np.sqrt(max(h_t, 1e-12))
            expo = float(np.sqrt(Omega) * (r[t] - mu_t) / h_sqrt + Delta)
            ll_sum += (0.5 * np.log(Omega)
                       + p * expo
                       - 0.5 * np.log(h_t)
                       - np.log(norm_const)
                       - (p + q) * np.log1p(np.exp(expo)))
        return float(-ll_sum / max(T, 1))

    def compute_aic_bic(self, log_likelihood_total: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC and BIC.

        Parameters
        ----------
        log_likelihood_total : float
            Total (summed) log-likelihood at optimum.
        num_params : int
            Number of estimated parameters.

        Returns
        -------
        (float, float)
            (AIC, BIC).
        """
        n = len(self.log_returns)
        aic = 2.0 * num_params - 2.0 * (log_likelihood_total)
        bic = np.log(n) * num_params - 2.0 * log_likelihood_total
        return float(aic), float(bic)

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], float, float, float, np.ndarray]:
        """
        Fit parameters via SLSQP with |beta|<1 constraint; bounds preserve original behavior.

        Parameters
        ----------
        initial_params : dict or None, default None
            If None, uses previous optimum else defaults:
            {'omega': var(r[:50])*(1-0.1-0.7), 'alpha':0.1, 'beta':0.7, 'p':3.5, 'q':3.5}.
        compute_metrics : bool, default False
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
                initial_params = {"omega": v0 * (1.0 - 0.1 - 0.7), "alpha": 0.1, "beta": 0.7, "p": 3.5, "q": 3.5}

        keys: List[str] = list(initial_params.keys())
        x0: List[float] = list(initial_params.values())

        bounds = [(-100.0, 100.0), (-10.0, 10.0), (0.0, 10.0), (0.001, 100.0), (0.001, 100.0)]
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
        Multi-step variance forecast via residual bootstrap.

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

        omega, alpha, beta, p, q = [self.optimal_params[k] for k in ["omega", "alpha", "beta", "p", "q"]]
        r = self.log_returns
        T = len(r)

        f = np.zeros(T, dtype=float)
        f[0] = np.log(np.var(r[:50]) if T > 50 else np.var(r))
        Delta = float(digamma(p) - digamma(q))
        Omega = float(polygamma(1, p) + polygamma(1, q))

        for t in range(1, T):
            rt = r[t - 1]
            h = float(np.exp(f[t - 1]))
            h_sqrt = np.sqrt(max(h, 1e-12))
            exp_arg = float(np.sqrt(Omega) * rt / h_sqrt + Delta)
            num = np.sqrt(Omega) * (-q - p) * rt * np.exp(exp_arg)
            den = 2.0 * h_sqrt * (np.exp(exp_arg) + 1.0)
            nabla = num / den - 0.5 - (np.sqrt(Omega) * p * rt) / (2.0 * h_sqrt)
            f[t] = omega + beta * f[t - 1] + alpha * nabla

        h = np.exp(f)
        z = r / np.sqrt(np.maximum(h, 1e-12))

        rT = r[-1]
        fT = f[-1]
        hT = float(np.exp(fT))
        hT_sqrt = np.sqrt(max(hT, 1e-12))
        exp_arg_T = float(np.sqrt(Omega) * rT / hT_sqrt + Delta)
        num_T = np.sqrt(Omega) * (-q - p) * rT * np.exp(exp_arg_T)
        den_T = 2.0 * hT_sqrt * (np.exp(exp_arg_T) + 1.0)
        nabla_T = num_T / den_T - 0.5 - (np.sqrt(Omega) * p * rT) / (2.0 * hT_sqrt)
        f1 = omega + beta * fT + alpha * nabla_T
        forecasts = [float(np.exp(f1))]
        f_prev = f1

        for _ in range(1, int(horizon)):
            zt = float(np.random.choice(z))
            h_prev = float(np.exp(f_prev))
            h_prev_sqrt = np.sqrt(max(h_prev, 1e-12))
            exp_arg = float(np.sqrt(Omega) * zt / h_prev_sqrt + Delta)
            num = np.sqrt(Omega) * (-q - p) * zt * np.exp(exp_arg)
            den = 2.0 * h_prev_sqrt * (np.exp(exp_arg) + 1.0)
            nab = num / den - 0.5 - (np.sqrt(Omega) * p * zt) / (2.0 * h_prev_sqrt)
            f_next = omega + beta * f_prev + alpha * nab
            forecasts.append(float(np.exp(f_next)))
            f_prev = f_next

        return np.array(forecasts, dtype=float)
