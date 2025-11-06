from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1, approx_fprime
from scipy.special import gamma


class LogLinearRealizedGARCHModelAST:
    """
    Log-linear Realized GARCH(1,1) with Asymmetric-Student-t (AST) innovations for returns
    and Gaussian measurement equation for the realized proxy (on the log scale).

    State (log variance):
        log h_t = omega + beta * log h_{t-1} + gamma * log x_{t-1}

    Returns block (AST):
        r_t | h_t ~ AST(delta, v1, v2) with standardized transform:
          z_t = (r_t - mu) / sqrt(h_t),  mu = 0
          w_t = m(delta,v1,v2) + s(delta,v1,v2) * z_t
          If w_t <= 0 ⇒ left tail (v1, alpha*), else right tail (v2, 1-alpha*)

        log f_r(r_t | h_t) =
            log s + log B(delta,v1,v2) - 0.5 * log h_t
            - (1 - I_t) * (v1+1)/2 * log(1 + (w_t / (2 alpha*))^2 / v1)
            -     I_t  * (v2+1)/2 * log(1 + (w_t / (2 (1-alpha*)))^2 / v2),

        with:
            K(v)   = Γ((v+1)/2) / (sqrt(pi v) Γ(v/2))
            B      = delta*K(v1) + (1-delta)*K(v2)
            alpha* = delta*K(v1)/B
            m      = 4 B [ -(alpha*^2) v1/(v1-1) + (1-alpha*)^2 v2/(v2-1) ]
            s      = sqrt( 4[ delta alpha*^2 v1/(v1-2) + (1-delta)(1-alpha*)^2 v2/(v2-2) ] - m^2 )

    Measurement (Gaussian on log x):
        u_t = log x_t - xi - phi * log h_t - (tau_1 z_t + tau_2(z_t^2 - 1)) ~ N(0, sigma_u^2)

    Parameters
    ----------
    log_returns : array_like
        Returns series (1-D).
    x : array_like
        Realized measure series (1-D, same length as returns).
    """

    model_name: str = "LLRGARCH-AST"
    distribution: str = "AST"

    def __init__(self, log_returns: np.ndarray, x: np.ndarray) -> None:
        r = np.asarray(log_returns, dtype=float).ravel()
        xr = np.asarray(x, dtype=float).ravel()
        if r.size != xr.size:
            raise ValueError("log_returns and x must have the same length.")
        if r.ndim != 1 or xr.ndim != 1:
            raise ValueError("Inputs must be 1-D arrays.")
        if np.any(xr <= 0):
            raise ValueError("All realized measures x must be positive (log is used).")

        self.log_returns: np.ndarray = r
        self.x: np.ndarray = xr

        self.optimal_params: Optional[Dict[str, float]] = None
        self.log_likelihood_value: Optional[float] = None  # total LL over T
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None
        self.standard_errors: Optional[np.ndarray] = None

    # ---------- AST helper functions ----------
    @staticmethod
    def _K(v: float) -> float:
        return float(gamma((v + 1.0) / 2.0) / (np.sqrt(np.pi * v) * gamma(v / 2.0)))

    def _B(self, delta: float, v1: float, v2: float) -> float:
        return float(delta * self._K(v1) + (1.0 - delta) * self._K(v2))

    def _alpha_star(self, delta: float, v1: float, v2: float) -> float:
        Bv = self._B(delta, v1, v2)
        return float(delta * self._K(v1) / Bv)

    def _m(self, delta: float, v1: float, v2: float) -> float:
        a = self._alpha_star(delta, v1, v2)
        Bv = self._B(delta, v1, v2)
        return float(4.0 * Bv * (-(a**2) * v1 / (v1 - 1.0) + (1.0 - a) ** 2 * v2 / (v2 - 1.0)))

    def _s(self, delta: float, v1: float, v2: float) -> float:
        a = self._alpha_star(delta, v1, v2)
        m = self._m(delta, v1, v2)
        inside = 4.0 * (delta * a**2 * v1 / (v1 - 2.0) + (1.0 - delta) * (1.0 - a) ** 2 * v2 / (v2 - 2.0)) - m**2
        return float(np.sqrt(inside))

    # ---------- Likelihood ----------
    def log_likelihood(self, par_list: List[float]) -> float:
        """
        Average *negative* log-likelihood to minimize.

        Order of `par_list`:
          [omega, beta, gamma, xi, phi, tau_1, tau_2, sigma_u, delta, v1, v2]
        """
        (
            omega, beta, gamma_r, xi, phi, tau_1, tau_2, sigma_u, delta, v1, v2
        ) = [float(x) for x in par_list]

        # Basic parameter checks
        if not (0.0 < sigma_u < np.inf) or not np.isfinite(sigma_u):
            return 1e10
        if not (0.0 < delta < 1.0):
            return 1e10
        if v1 <= 2.0 or v2 <= 2.0:
            return 1e10
        # mild stability encouragement on beta (not strictly required, but helpful)
        if not np.isfinite(beta) or not np.isfinite(gamma_r) or not np.isfinite(phi):
            return 1e10

        r = self.log_returns
        x = self.x
        T = r.size
        if T < 2:
            return 1e10

        # Rebuild log variance
        log_h = np.empty(T, dtype=float)
        log_h[0] = float(np.log(np.var(r[: min(50, max(2, T))])))
        for t in range(1, T):
            log_h[t] = omega + beta * log_h[t - 1] + gamma_r * np.log(x[t - 1])

        h = np.exp(log_h)
        if not np.all(np.isfinite(h)) or np.any(h <= 0):
            return 1e10

        # AST constants (constant in t)
        Bv = self._B(delta, v1, v2)
        a_star = self._alpha_star(delta, v1, v2)
        m_ast = self._m(delta, v1, v2)
        s_ast = self._s(delta, v1, v2)
        if not np.all(np.isfinite([Bv, a_star, m_ast, s_ast])) or s_ast <= 0 or Bv <= 0:
            return 1e10

        # Standardized residuals for returns
        z = r / np.sqrt(h)
        w = m_ast + s_ast * z
        I = (w > 0).astype(float)

        # AST log-density for returns
        # term1 and term2 are computed vectorized
        denom_left = 2.0 * a_star
        denom_right = 2.0 * (1.0 - a_star)

        term_left = ((v1 + 1.0) / 2.0) * np.log1p((w / denom_left) ** 2 / v1)
        term_right = ((v2 + 1.0) / 2.0) * np.log1p((w / denom_right) ** 2 / v2)

        ll_r = (
            np.log(s_ast)
            + np.log(Bv)
            - 0.5 * log_h
            - (1.0 - I) * term_left
            - I * term_right
        )

        # Measurement equation on log x
        tau = tau_1 * z + tau_2 * (z**2 - 1.0)
        u = np.log(x) - xi - phi * log_h - tau
        ll_x = -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma_u) + (u**2) / (sigma_u**2))

        ll = ll_r + ll_x
        if not np.all(np.isfinite(ll)):
            return 1e10

        return float(-np.mean(ll))

    # ---------- IC ----------
    def compute_aic_bic(self, total_ll: float, num_params: int) -> Tuple[float, float]:
        T = self.log_returns.size
        aic = 2.0 * num_params - 2.0 * total_ll
        bic = np.log(T) * num_params - 2.0 * total_ll
        return float(aic), float(bic)

    # ---------- Estimation ----------
    def optimize(self,initial_params: Optional[Dict[str, float]] = None,compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        MLE via SLSQP with sensible bounds.

        initial_params keys:
          omega, beta, gamma, xi, phi, tau_1, tau_2, sigma_u, delta, v1, v2
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = dict(
                    omega=0.95,
                    beta=0.30,
                    gamma=0.40,
                    xi=np.log(self.x.mean()) - 0.5,  # heuristic
                    phi=1.20,
                    tau_1=-0.05,
                    tau_2=0.08,
                    sigma_u=0.40,
                    delta=0.5,
                    v1=4.5,
                    v2=4.5,
                )

        keys = list(initial_params.keys())
        x0 = [float(initial_params[k]) for k in keys]

        bounds = [
            (-np.inf, np.inf),  # omega
            (-0.999, 0.999),    # beta (soft stationarity)
            (0.0, np.inf),      # gamma >= 0
            (-np.inf, np.inf),  # xi
            (0.0, np.inf),      # phi >= 0
            (-np.inf, np.inf),  # tau_1
            (-np.inf, np.inf),  # tau_2
            (1e-5, np.inf),     # sigma_u > 0
            (1e-5, 1.0 - 1e-5), # delta in (0,1)
            (2.001, np.inf),    # v1 > 2
            (2.001, np.inf),    # v2 > 2
        ]

        res = opt.minimize(self.log_likelihood, x0, method="SLSQP", bounds=bounds, options={"maxiter": 500})
        self.convergence = bool(res.success)

        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")
            # Keep previous MLE (if any) or return starting values
            return self.optimal_params if self.optimal_params is not None else dict(zip(keys, x0))

        if compute_metrics:
            T = self.log_returns.size
            total_ll = float(-res.fun * T)  # res.fun is average negative ll
            self.log_likelihood_value = total_ll
            k = len(keys)
            self.aic, self.bic = self.compute_aic_bic(total_ll, k)

            # Standard errors from Hessian of avg neg-ll scaled by T
            par_vec = np.array([self.optimal_params[k] for k in keys], dtype=float)
            H_avg = approx_hess1(par_vec, self.log_likelihood)
            H = H_avg * T
            try:
                cov = inv(H)
                ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                ses = np.full(k, np.nan)

            if np.isnan(ses).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                g = approx_fprime(par_vec, self.log_likelihood, eps)
                try:
                    cov_fb = inv(np.outer(g, g) * T)
                    ses = np.sqrt(np.maximum(np.diag(cov_fb), 0.0))
                except np.linalg.LinAlgError:
                    ses = np.full(k, np.nan)

            self.standard_errors = ses

            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors

        return self.optimal_params

    # ---------- Forecasting ----------
    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Forecast h_{T+1}, …, h_{T+h} assuming log-linear recursion and x_t held at last observed value.

        Returns
        -------
        np.ndarray
            Variance forecasts on the original scale.
        """
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")

        omega = float(self.optimal_params["omega"])
        beta = float(self.optimal_params["beta"])
        gamma_r = float(self.optimal_params["gamma"])

        r = self.log_returns
        x = self.x
        T = r.size

        # Rebuild log_h in-sample
        log_h = np.empty(T, dtype=float)
        log_h[0] = float(np.log(np.var(r[: min(50, max(2, T))])))
        for t in range(1, T):
            log_h[t] = omega + beta * log_h[t - 1] + gamma_r * np.log(x[t - 1])

        log_h_last = log_h[-1]
        x_last = float(x[-1])

        # Forward iterate
        fcst = []
        prev = log_h_last
        lx = float(np.log(x_last))
        for _ in range(int(horizon)):
            nxt = omega + beta * prev + gamma_r * lx
            fcst.append(np.exp(nxt))
            prev = nxt

        return np.asarray(fcst, dtype=float)
