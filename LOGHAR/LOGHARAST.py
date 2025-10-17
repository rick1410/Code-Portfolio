from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1, approx_fprime
from scipy.special import gamma


class LogHARASTModel:
    """
    Log-HAR model with Asymmetric Student-t (AST) innovations on log-realized volatility.

    The observation is y_t = log(RV_t). The conditional mean follows the HAR structure in logs,
    and residuals are assumed to follow an AST density with parameters (delta, v1, v2) and
    scale sigma^2. See score/equations in implementation below.

    Parameters
    ----------
    realized_volatility : np.ndarray
        Positive realized volatility series; internally we use y_t = log(realized_volatility).

    Attributes
    ----------
    model_name : str
        Human-readable identifier.
    distribution : str
        Name of the innovation distribution.
    only_kernel : bool
        Marker used by some pipelines.
    optimal_params : Optional[Dict[str, float]]
        Dictionary of fitted parameters if `optimize` has succeeded.
    log_likelihood_value : Optional[float]
        Total (not averaged) log-likelihood at optimum (using effective sample size).
    aic, bic : Optional[float]
        Information criteria computed from total log-likelihood.
    convergence : Optional[bool]
        True if the optimizer reported success.
    standard_errors : Optional[np.ndarray]
        Standard errors derived from the inverse Hessian (may contain NaNs if inversion fails).
    """

    model_name: str = "LHAR-AST"
    distribution: str = "AST"
    only_kernel: bool = True

    def __init__(self, realized_volatility: np.ndarray) -> None:
        rv = np.asarray(realized_volatility, dtype=float)
        if np.any(rv <= 0):
            raise ValueError("All realized_volatility values must be positive to take logs.")
        self.realized_volatility = rv
        self.log_rv = np.log(rv)

        self.optimal_params: Optional[Dict[str, float]] = None
        self.log_likelihood_value: Optional[float] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None
        self.standard_errors: Optional[np.ndarray] = None

    # ===== AST helpers =====
    @staticmethod
    def _K(v: float) -> float:
        return float(gamma((v + 1.0) / 2.0) / (np.sqrt(np.pi * v) * gamma(v / 2.0)))

    def _B(self, delta: float, v1: float, v2: float) -> float:
        return delta * self._K(v1) + (1.0 - delta) * self._K(v2)

    def _alpha_star(self, delta: float, v1: float, v2: float) -> float:
        Bv = self._B(delta, v1, v2)
        return float((delta * self._K(v1)) / Bv)

    def _m_ast(self, delta: float, v1: float, v2: float) -> float:
        a = self._alpha_star(delta, v1, v2)
        Bv = self._B(delta, v1, v2)
        return float(4.0 * Bv * (-(a ** 2) * (v1 / (v1 - 1.0)) + (1.0 - a) ** 2 * (v2 / (v2 - 1.0))))

    def _s_ast(self, delta: float, v1: float, v2: float) -> float:
        a = self._alpha_star(delta, v1, v2)
        Bv = self._B(delta, v1, v2)
        m_val = self._m_ast(delta, v1, v2)
        inside = 4.0 * (delta * a ** 2 * (v1 / (v1 - 2.0)) + (1.0 - delta) * (1.0 - a) ** 2 * (v2 / (v2 - 2.0))) - m_val ** 2
        if inside <= 0:
            # invalid parameter region produces non-positive variance scale
            return np.nan
        return float(np.sqrt(inside))

    @staticmethod
    def _I(x: float) -> int:
        return 1 if x > 0.0 else 0

    # ===== HAR mean on log scale =====
    @staticmethod
    def _har_mean_equation(t: int, y: np.ndarray, alpha0: float, alpha1: float, alpha2: float, alpha3: float) -> float:
        daily = y[t - 1] if t >= 1 else y[0]
        weekly = float(np.mean(y[max(0, t - 5):t])) if t > 0 else y[0]
        monthly = float(np.mean(y[max(0, t - 22):t])) if t > 0 else y[0]
        return float(alpha0 + alpha1 * daily + alpha2 * weekly + alpha3 * monthly)

    # ===== Likelihood =====
    def log_likelihood(self, params: List[float]) -> float:
        """
        Average negative log-likelihood for Log-HAR with AST residuals.

        Parameters
        ----------
        params : list
            [alpha0, alpha1, alpha2, alpha3, sigma2, delta, v1, v2]

        Returns
        -------
        float
            Average negative log-likelihood (to minimize).
        """
        alpha0, alpha1, alpha2, alpha3, sigma2, delta, v1, v2 = params

        # Basic parameter checks
        if sigma2 <= 0 or not (0 < delta < 1) or v1 <= 2 or v2 <= 2:
            return 1e10

        y = self.log_rv
        T = len(y)
        if T < 23:
            # Need at least 23 obs to form 22-day lag block + 1
            return 1e10

        m_val = self._m_ast(delta, v1, v2)
        s_val = self._s_ast(delta, v1, v2)
        if not np.isfinite(s_val) or s_val <= 0:
            return 1e10
        a_star = self._alpha_star(delta, v1, v2)
        Bv = self._B(delta, v1, v2)

        residuals = np.array([y[t] - self._har_mean_equation(t, y, alpha0, alpha1, alpha2, alpha3)
                              for t in range(22, T)], dtype=float)
        T_eff = residuals.size
        if T_eff == 0:
            return 1e10

        # AST log-likelihood for residuals e_t ~ AST(m(m,delta,v1,v2) + s * (e_t/sqrt(sigma2)))
        scaled = residuals / np.sqrt(sigma2)
        z = m_val + s_val * scaled

        # piecewise for z sign
        I_pos = (z > 0).astype(float)
        I_neg = 1.0 - I_pos

        term1 = ((v1 + 1.0) / 2.0) * np.log1p((z / (2.0 * a_star)) ** 2 / v1)
        term2 = ((v2 + 1.0) / 2.0) * np.log1p((z / (2.0 * (1.0 - a_star))) ** 2 / v2)

        ll = np.log(s_val) + np.log(Bv) - 0.5 * np.log(sigma2) - I_neg * term1 - I_pos * term2
        return float(-np.mean(ll))

    # ===== ICs =====
    def compute_aic_bic(self, total_ll: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC/BIC using total log-likelihood and number of parameters.
        """
        n = len(self.log_rv)
        aic = 2.0 * num_params - 2.0 * total_ll
        bic = np.log(n) * num_params - 2.0 * total_ll
        return float(aic), float(bic)

    # ===== Estimation =====
    def optimize(self,initial_params: Optional[Dict[str, float]] = None,compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Estimate parameters by SLSQP with sensible bounds.

        Parameters
        ----------
        initial_params : dict or None
            Dict with keys: alpha0, alpha1, alpha2, alpha3, sigma2, delta, v1, v2.
        compute_metrics : bool
            If True, also return (AIC, BIC, total_LL, SEs).

        Returns
        -------
        dict or tuple
            Optimal params dict; or (params, AIC, BIC, total_LL, SEs) if `compute_metrics=True`.
        """
        y = self.log_rv

        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {
                    "alpha0": float(np.mean(y)),
                    "alpha1": 0.5,
                    "alpha2": 0.2,
                    "alpha3": 0.1,
                    "sigma2": float(np.var(y) * 0.5 if y.size > 1 else 1e-3),
                    "delta": 0.5,
                    "v1": 4.0,
                    "v2": 4.0,
                }

        keys = list(initial_params.keys())
        x0 = [float(initial_params[k]) for k in keys]

        bounds = [
            (-10.0, 10.0),   # alpha0
            (-10.0, 10.0),   # alpha1
            (-10.0, 10.0),   # alpha2
            (-10.0, 10.0),   # alpha3
            (1e-12, 1e12),   # sigma2
            (1e-8, 1.0 - 1e-8),  # delta
            (2.001, 200.0),  # v1
            (2.001, 200.0),  # v2
        ]

        obj = lambda arr: self.log_likelihood(arr)
        res = opt.minimize(obj, x0, method="SLSQP", bounds=bounds)

        self.convergence = bool(res.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")
            return self.optimal_params if self.optimal_params is not None else initial_params

        if compute_metrics and self.convergence:
            # res.fun is average negative log-likelihood → convert to total LL with effective T
            T_eff = max(0, len(y) - 22)
            total_ll = float(-res.fun * T_eff)
            self.log_likelihood_value = total_ll

            k = len(keys)
            self.aic, self.bic = self.compute_aic_bic(total_ll, k)

            # SEs from Hessian of average neg-LL → scale by T_eff
            par_vec = np.array([self.optimal_params[k] for k in keys], dtype=float)
            H_avg = approx_hess1(par_vec, self.log_likelihood)  # Hessian of avg neg-LL
            H = H_avg * max(T_eff, 1)  # approximate Hessian of total neg-LL
            try:
                cov = inv(H)
                ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                ses = np.full(k, np.nan)

            if np.isnan(ses).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                g = approx_fprime(par_vec, self.log_likelihood, eps)
                try:
                    cov_fallback = inv(np.outer(g, g) * max(T_eff, 1))
                    ses = np.sqrt(np.maximum(np.diag(cov_fallback), 0.0))
                except np.linalg.LinAlgError:
                    ses = np.full(k, np.nan)

            self.standard_errors = ses
            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors

        return self.optimal_params

    # ===== Forecasting =====
    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Multi-step-ahead forecasts of realized volatility (back on original scale).

        Forecasts are generated on the log scale using the HAR recursion and then exponentiated.

        Parameters
        ----------
        horizon : int
            Number of steps to forecast.

        Returns
        -------
        np.ndarray
            Forecasts for RV_{T+1}, ..., RV_{T+horizon}.
        """
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")

        alpha0 = float(self.optimal_params["alpha0"])
        alpha1 = float(self.optimal_params["alpha1"])
        alpha2 = float(self.optimal_params["alpha2"])
        alpha3 = float(self.optimal_params["alpha3"])

        hist = list(self.log_rv.astype(float))
        log_fcst: List[float] = []

        for _ in range(int(horizon)):
            daily = hist[-1]
            weekly = float(np.mean(hist[-5:])) if len(hist) >= 5 else float(np.mean(hist))
            monthly = float(np.mean(hist[-22:])) if len(hist) >= 22 else float(np.mean(hist))
            lf = alpha0 + alpha1 * daily + alpha2 * weekly + alpha3 * monthly
            log_fcst.append(lf)
            hist.append(lf)

        return np.exp(np.asarray(log_fcst, dtype=float))
