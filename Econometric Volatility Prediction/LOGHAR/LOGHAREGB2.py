from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1, approx_fprime
from scipy.special import gamma, digamma, polygamma


class LogHAREGB2Model:
    """
    Log-HAR model with EGB2 innovations on log-realized volatility.

    We model y_t = log(RV_t). The conditional mean of y_t follows a HAR structure:
        E[y_t | F_{t-1}] = alpha0 + alpha1 * y_{t-1}
                           + alpha2 * mean(y_{t-5..t-1})
                           + alpha3 * mean(y_{t-22..t-1})

    Residuals e_t = y_t - E[y_t|F_{t-1}] are assumed EGB2 with parameters (p, q) and
    scale sigma^2. The EGB2 log-likelihood is used for estimation.

    Parameters
    ----------
    realized_volatility : array_like
        Positive realized volatility series; we internally use its log.
    """

    model_name: str = "LHAR-EGB2"
    distribution: str = "EGB2"
    only_kernel: bool = True  # uses realized-volatility-only "kernel" data

    def __init__(self, realized_volatility: np.ndarray) -> None:
        rv = np.asarray(realized_volatility, dtype=float)
        if np.any(rv <= 0):
            raise ValueError("All realized_volatility values must be positive to take logs.")
        self.realized_volatility: np.ndarray = rv
        self.log_rv: np.ndarray = np.log(rv)

        # Fit artifacts
        self.optimal_params: Optional[Dict[str, float]] = None
        self.log_likelihood_value: Optional[float] = None  # total LL (effective T)
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None
        self.standard_errors: Optional[np.ndarray] = None

    # ===== Mean equation (HAR on logs) =====
    @staticmethod
    def _har_mean_equation(t: int, y: np.ndarray, a0: float, a1: float, a2: float, a3: float) -> float:
        daily = y[t - 1]
        weekly = float(np.mean(y[max(0, t - 5):t]))
        monthly = float(np.mean(y[max(0, t - 22):t]))
        return float(a0 + a1 * daily + a2 * weekly + a3 * monthly)

    # ===== EGB2 helpers =====
    @staticmethod
    def _egb2_delta_omega(p: float, q: float) -> Tuple[float, float]:
        """Return Delta = ψ(p) - ψ(q), Omega = ψ'(p) + ψ'(q)."""
        Delta = float(digamma(p) - digamma(q))
        Omega = float(polygamma(1, p) + polygamma(1, q))
        return Delta, Omega

    @staticmethod
    def _egb2_log_norm_const(p: float, q: float) -> float:
        """log normalization constant log(Γ(p)Γ(q)/Γ(p+q))."""
        return float(np.log(gamma(p) * gamma(q) / gamma(p + q)))

    # ===== Likelihood =====
    def log_likelihood(self, params: List[float]) -> float:
        """
        Average negative log-likelihood for Log-HAR with EGB2 residuals.

        Parameters
        ----------
        params : list
            [alpha0, alpha1, alpha2, alpha3, sigma2, p, q]

        Returns
        -------
        float
            Average negative log-likelihood (to minimize).
        """
        alpha0, alpha1, alpha2, alpha3, sigma2, p, q = params

        # Basic parameter checks
        if sigma2 <= 0 or p <= 0 or q <= 0:
            return 1e10

        y = self.log_rv
        T = len(y)
        if T < 23:
            # need 22 days of lags + one observation
            return 1e10

        Delta, Omega = self._egb2_delta_omega(p, q)
        if not np.isfinite(Omega) or Omega <= 0:
            return 1e10
        log_c = self._egb2_log_norm_const(p, q)

        # Build residuals from t=22 onward (first 22 obs form lags)
        resid = np.array([y[t] - self._har_mean_equation(t, y, alpha0, alpha1, alpha2, alpha3)
                          for t in range(22, T)], dtype=float)
        T_eff = resid.size
        if T_eff == 0:
            return 1e10

        scaled = resid / np.sqrt(sigma2)
        z = np.sqrt(Omega) * scaled + Delta

        ll_terms = (
            0.5 * np.log(Omega)
            + p * z
            - log_c
            - (p + q) * np.log1p(np.exp(z))
            - 0.5 * np.log(sigma2)
        )

        # Return average *negative* log-likelihood
        return float(-np.mean(ll_terms))

    # ===== ICs =====
    def compute_aic_bic(self, total_ll: float, num_params: int) -> Tuple[float, float]:
        """
        Compute AIC/BIC from total (effective-sample) log-likelihood.
        """
        n = len(self.log_rv)
        aic = 2.0 * num_params - 2.0 * total_ll
        bic = np.log(n) * num_params - 2.0 * total_ll
        return float(aic), float(bic)

    # ===== Estimation =====
    def optimize(self,initial_params: Optional[Dict[str, float]] = None,compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Estimate parameters by SLSQP subject to simple bounds.

        Parameters
        ----------
        initial_params : dict or None
            Keys: alpha0, alpha1, alpha2, alpha3, sigma2, p, q.
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
                    "p": 3.5,
                    "q": 3.5,
                }

        keys = list(initial_params.keys())
        x0 = [float(initial_params[k]) for k in keys]

        bounds = [
            (-10.0, 10.0),   # alpha0
            (-10.0, 10.0),   # alpha1
            (-10.0, 10.0),   # alpha2
            (-10.0, 10.0),   # alpha3
            (1e-12, 1e12),   # sigma2
            (1e-3, 100.0),   # p
            (1e-3, 100.0),   # q
        ]

        res = opt.minimize(self.log_likelihood, x0, method="SLSQP", bounds=bounds)
        self.convergence = bool(res.success)

        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")
            return self.optimal_params if self.optimal_params is not None else dict(zip(keys, x0))

        if compute_metrics and self.convergence:
            # res.fun is average negative log-likelihood over effective sample (T_eff)
            T_eff = max(0, len(y) - 22)
            total_ll = float(-res.fun * T_eff)
            self.log_likelihood_value = total_ll

            k = len(keys)
            self.aic, self.bic = self.compute_aic_bic(total_ll, k)

            # Standard errors via Hessian of average neg-LL scaled by T_eff
            par_vec = np.array([self.optimal_params[k] for k in keys], dtype=float)
            H_avg = approx_hess1(par_vec, self.log_likelihood)
            H = H_avg * max(T_eff, 1)

            try:
                cov = inv(H)
                ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                ses = np.full(k, np.nan)

            if np.isnan(ses).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                g = approx_fprime(par_vec, self.log_likelihood, eps)
                try:
                    cov_fb = inv(np.outer(g, g) * max(T_eff, 1))
                    ses = np.sqrt(np.maximum(np.diag(cov_fb), 0.0))
                except np.linalg.LinAlgError:
                    ses = np.full(k, np.nan)

            self.standard_errors = ses
            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors

        return self.optimal_params

    # ===== Forecasts =====
    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Multi-step-ahead forecasts of realized volatility (on original scale).

        We iterate the HAR recursion on the log scale and then exponentiate.

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

        a0 = float(self.optimal_params["alpha0"])
        a1 = float(self.optimal_params["alpha1"])
        a2 = float(self.optimal_params["alpha2"])
        a3 = float(self.optimal_params["alpha3"])

        hist = list(self.log_rv.astype(float))
        log_fcst: List[float] = []

        for _ in range(int(horizon)):
            daily = hist[-1]
            weekly = float(np.mean(hist[-5:])) if len(hist) >= 5 else float(np.mean(hist))
            monthly = float(np.mean(hist[-22:])) if len(hist) >= 22 else float(np.mean(hist))
            lf = a0 + a1 * daily + a2 * weekly + a3 * monthly
            log_fcst.append(lf)
            hist.append(lf)

        return np.exp(np.asarray(log_fcst, dtype=float))
