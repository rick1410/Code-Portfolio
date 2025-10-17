from typing import Dict, Optional, Sequence, Tuple, List
import numpy as np
import scipy
from scipy.special import gamma
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1


class ASTGASModel:
    """
    GAS model with Asymmetric Student-t (AST) innovations.

    Parameters
    ----------
    log_returns : np.ndarray
        1-D array of returns.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    distribution : str
        Innovation distribution ("AST").
    log_returns : np.ndarray
        Return series (float64).
    optimal_params : Optional[Dict[str, float]]
        Fitted parameter dictionary with keys {"omega","alpha","beta","delta","v1","v2"}.
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

    model_name: str = "GAS-AST"
    distribution: str = "AST"

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

    def K(self, v: float) -> float:
        """
        Normalizing K(v) = Γ((v+1)/2) / (sqrt(pi*v) Γ(v/2)).

        Parameters
        ----------
        v : float
            Degrees of freedom (v > 0).

        Returns
        -------
        float
            K(v).
        """
        return float(gamma((v + 1.0) / 2.0) / (np.sqrt(np.pi * v) * gamma(v / 2.0)))

    def B(self, delta: float, v1: float, v2: float) -> float:
        """
        Mixture weight B(delta,v1,v2) = δ K(v1) + (1-δ) K(v2).

        Parameters
        ----------
        delta : float
            Asymmetry weight in [0,1].
        v1 : float
            Left-tail degrees of freedom (> 0).
        v2 : float
            Right-tail degrees of freedom (> 0).

        Returns
        -------
        float
            B(delta,v1,v2).
        """
        return float(delta * self.K(v1) + (1.0 - delta) * self.K(v2))

    def alpha_star(self, delta: float, v1: float, v2: float) -> float:
        """
        Alpha-star for AST.

        Parameters
        ----------
        delta : float
            Asymmetry weight.
        v1 : float
            Left-tail dof.
        v2 : float
            Right-tail dof.

        Returns
        -------
        float
            α* in (0,1).
        """
        Bv = self.B(delta, v1, v2)
        return float((delta * self.K(v1)) / Bv)

    def m(self, delta: float, v1: float, v2: float) -> float:
        """
        Location adjustment m for AST.

        Parameters
        ----------
        delta : float
        v1 : float
        v2 : float

        Returns
        -------
        float
            m(delta,v1,v2).
        """
        a = self.alpha_star(delta, v1, v2)
        Bv = self.B(delta, v1, v2)
        return float(4.0 * Bv * (-(a**2) * v1 / (v1 - 1.0) + (1.0 - a) ** 2 * v2 / (v2 - 1.0)))

    def s(self, delta: float, v1: float, v2: float) -> float:
        """
        Scale adjustment s for AST.

        Parameters
        ----------
        delta : float
        v1 : float
        v2 : float

        Returns
        -------
        float
            s(delta,v1,v2) > 0.
        """
        a = self.alpha_star(delta, v1, v2)
        m_val = self.m(delta, v1, v2)
        val = 4.0 * (delta * a**2 * v1 / (v1 - 2.0) + (1.0 - delta) * (1.0 - a) ** 2 * v2 / (v2 - 2.0)) - m_val**2
        return float(np.sqrt(val))

    def I_t(self, r_t: float, mu_t: float, h_t: float, m_val: float, s_val: float) -> int:
        """
        Indicator I_t = 1{ m + s * (r_t - mu_t) / sqrt(h_t) > 0 }.

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
            1 or 0.
        """
        return 1 if m_val + s_val * (r_t - mu_t) / np.sqrt(max(h_t, 1e-12)) > 0.0 else 0

    def nabla_t(self, r_t: float, mu_t: float, h_t: float, delta: float, v1: float, v2: float) -> float:
        """
        Score with respect to log-variance state f_t.

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
            Score ∇_t.
        """
        h_sqrt = np.sqrt(max(h_t, 1e-12))
        m_val = self.m(delta, v1, v2)
        s_val = self.s(delta, v1, v2)
        a = self.alpha_star(delta, v1, v2)
        It = self.I_t(r_t, mu_t, h_t, m_val, s_val)
        num = s_val * (r_t - mu_t) / h_sqrt + m_val

        denom2 = 4.0 * (1.0 - a) ** 2 * v2 * (num**2 / (4.0 * (1.0 - a) ** 2 * v2) + 1.0) * h_sqrt
        denom1 = 4.0 * a**2 * v1 * (num**2 / (4.0 * a**2 * v1) + 1.0) * h_sqrt

        term1 = num / denom2
        term2 = num / denom1
        return float(-0.5 + It * (v2 + 1.0) / 2.0 * term1 + (1.0 - It) * (v1 + 1.0) / 2.0 * term2)

    def log_likelihood(self, params: Sequence[float]) -> float:
        """
        Average negative log-likelihood for GAS-AST (to minimize).

        Parameters
        ----------
        params : sequence of float
            (omega, alpha, beta, delta, v1, v2).

        Returns
        -------
        float
            Average negative log-likelihood.
        """
        omega, alpha, beta, delta, v1, v2 = params
        T = len(self.log_returns)
        mu_t = 0.0

        f = np.zeros(T, dtype=float)
        f[0] = np.log(np.var(self.log_returns[:50]) if T > 50 else np.var(self.log_returns))
        for t in range(1, T):
            h_tm1 = float(np.exp(f[t - 1]))
            grad = self.nabla_t(self.log_returns[t - 1], mu_t, h_tm1, delta, v1, v2)
            f[t] = omega + beta * f[t - 1] + alpha * grad

        ll = np.zeros(T, dtype=float)
        for t in range(T):
            h_t = float(np.exp(f[t]))
            m_val = self.m(delta, v1, v2)
            s_val = self.s(delta, v1, v2)
            a = self.alpha_star(delta, v1, v2)
            Bv = self.B(delta, v1, v2)
            It = self.I_t(self.log_returns[t], mu_t, h_t, m_val, s_val)

            z = (m_val + s_val * (self.log_returns[t] - mu_t) / np.sqrt(max(h_t, 1e-12)))
            term1 = (v1 + 1.0) / 2.0 * np.log(1.0 + (z / (2.0 * a)) ** 2 / v1)
            term2 = (v2 + 1.0) / 2.0 * np.log(1.0 + (z / (2.0 * (1.0 - a))) ** 2 / v2)
            ll[t] = np.log(s_val) + np.log(Bv) - 0.5 * np.log(h_t) - (1.0 - It) * term1 - It * term2

        return float(-np.mean(ll))

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
            {'omega':0.057,'alpha':0.203,'beta':0.979,'delta':0.5,'v1':2.4,'v2':2.4}.
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
                initial_params = {"omega": 0.057, "alpha": 0.203, "beta": 0.979, "delta": 0.5, "v1": 2.4, "v2": 2.4}

        keys: List[str] = list(initial_params.keys())
        x0: List[float] = list(initial_params.values())

        bounds = [(-100.0, 100.0), (-10.0, 10.0), (0.0, 10.0), (0.0, 1.0), (0.001, None), (0.001, None)]
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
            hessian_inv = inv(hessian) / len(self.log_returns)
            self.standard_errors = np.sqrt(np.maximum(np.diag(hessian_inv), 0.0))

            if np.isnan(self.standard_errors).any():
                epsilon = float(np.sqrt(np.finfo(float).eps))
                grad = scipy.optimize.approx_fprime(list(self.optimal_params.values()), self.log_likelihood, epsilon)
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
        if self.optimal_params is None: raise RuntimeError("Optimize model before forecasting.")

        omega, alpha, beta, delta, v1, v2 = [self.optimal_params[k] for k in ["omega", "alpha", "beta", "delta", "v1", "v2"]]
        mu = 0.0

        T = len(self.log_returns)
        f = np.zeros(T, dtype=float)
        f[0] = np.log(np.var(self.log_returns[:50]) if T > 50 else np.var(self.log_returns))
        for t in range(1, T):
            h_tm1 = float(np.exp(f[t - 1]))
            grad = self.nabla_t(self.log_returns[t - 1], mu, h_tm1, delta, v1, v2)
            f[t] = omega + beta * f[t - 1] + alpha * grad

        h = np.exp(f)
        z = self.log_returns / np.sqrt(np.maximum(h, 1e-12))

        rT = self.log_returns[-1]
        fT = f[-1]
        hT = float(np.exp(fT))
        grad1 = self.nabla_t(rT, mu, hT, delta, v1, v2)
        f1 = omega + beta * fT + alpha * grad1
        forecasts = [float(np.exp(f1))]
        f_prev = f1

        for _ in range(1, int(horizon)):
            zt = float(np.random.choice(z))
            hprev = float(np.exp(f_prev))
            eps = zt * np.sqrt(hprev)
            grad_next = self.nabla_t(eps, mu, hprev, delta, v1, v2)
            f_next = omega + beta * f_prev + alpha * grad_next
            forecasts.append(float(np.exp(f_next)))
            f_prev = f_next

        return np.array(forecasts, dtype=float)
