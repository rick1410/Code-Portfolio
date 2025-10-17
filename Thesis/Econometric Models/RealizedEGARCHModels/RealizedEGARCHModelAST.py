from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1, approx_fprime
from scipy.special import gamma


class RealizedEGARCHModelAST:
    """
    Realized EGARCH(1,1) with Asymmetric-Student-t (AST) returns and
    Gaussian measurement equation on log(x).

    State (log variance):
        log h_t = omega + beta * log h_{t-1} + tau_{t-1} + delta1 * u_{t-1}
        where tau_{t} = tau_1 * z_t + tau_2 * (z_t^2 - 1),
              z_t = r_t / sqrt(h_t),
              u_t = log x_t - xi - log h_t.

    Returns block (AST with parameters delta in (0,1), v1>2, v2>2):
      Uses the parametrization in which
        K(v) = Γ((v+1)/2)/(sqrt(pi*v)*Γ(v/2))
        B = delta*K(v1) + (1-delta)*K(v2)
        alpha* = (delta*K(v1)) / B
        m, s follow standard AST centering/scaling definitions.

      The per-observation log-likelihood is the piecewise AST log-kernel
      (vectorized via indicator I(z)>0).

    Measurement (Gaussian on log x):
        u_t ~ N(0, sigma_u^2)
    """

    model_name: str = "REGARCH-AST"
    distribution: str = "AST"

    # ---------- init ----------
    def __init__(self, log_returns: np.ndarray, x: np.ndarray) -> None:
        r = np.asarray(log_returns, dtype=float).ravel()
        xr = np.asarray(x, dtype=float).ravel()
        if r.size != xr.size:
            raise ValueError("log_returns and x must have the same length.")
        if np.any(xr <= 0):
            raise ValueError("All realized measures x must be positive (log is used).")

        self.log_returns: np.ndarray = r
        self.x: np.ndarray = xr

        self.optimal_params: Optional[Dict[str, float]] = None
        self.log_likelihood_value: Optional[float] = None  # total over T
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None
        self.standard_errors: Optional[np.ndarray] = None

    # ---------- AST helpers ----------
    @staticmethod
    def _K(v: float) -> float:
        return float(gamma((v + 1.0) / 2.0) / (np.sqrt(np.pi * v) * gamma(v / 2.0)))

    def _B(self, delta: float, v1: float, v2: float) -> float:
        return delta * self._K(v1) + (1.0 - delta) * self._K(v2)

    def _alpha_star(self, delta: float, v1: float, v2: float) -> float:
        Bv = self._B(delta, v1, v2)
        return float(delta * self._K(v1) / Bv)

    def _m_ast(self, delta: float, v1: float, v2: float) -> float:
        a = self._alpha_star(delta, v1, v2)
        Bv = self._B(delta, v1, v2)
        return float(4.0 * Bv * (-(a ** 2) * (v1 / (v1 - 1.0)) + (1.0 - a) ** 2 * (v2 / (v2 - 1.0))))

    def _s_ast(self, delta: float, v1: float, v2: float) -> float:
        a = self._alpha_star(delta, v1, v2)
        Bv = self._B(delta, v1, v2)
        m = self._m_ast(delta, v1, v2)
        inside = 4.0 * (delta * a ** 2 * (v1 / (v1 - 2.0)) + (1.0 - delta) * (1.0 - a) ** 2 * (v2 / (v2 - 2.0))) - m ** 2
        return float(np.sqrt(inside))

    # ---------- likelihood ----------
    def _avg_neg_loglik(self, par_list: List[float]) -> float:
        """
        Average *negative* log-likelihood to minimize.

        Parameter order (natural scale):
          [omega, beta, delta1, xi, tau_1, tau_2, sigma_u, delta, v1, v2]
          with constraints: sigma_u>0, delta in (0,1), v1>2, v2>2, beta<1 (enforced via constraint).
        """
        omega, beta, delta1, xi, tau_1, tau_2, sigma_u, delt, v1, v2 = [float(v) for v in par_list]

        # Basic checks
        if sigma_u <= 0 or not np.isfinite(sigma_u):
            return 1e10
        if not (1e-8 < delt < 1.0 - 1e-8):
            return 1e10
        if v1 <= 2.0 or v2 <= 2.0:
            return 1e10
        if not np.all(np.isfinite([omega, beta, delta1, xi, tau_1, tau_2])):
            return 1e10

        r = self.log_returns
        x = self.x
        T = r.size
        if T < 2:
            return 1e10

        # Precompute AST constants (do not depend on t)
        Bv = self._B(delt, v1, v2)
        a_star = self._alpha_star(delt, v1, v2)
        m = self._m_ast(delt, v1, v2)
        s = self._s_ast(delt, v1, v2)

        # Build state and residuals
        log_h = np.empty(T, dtype=float)
        z = np.empty(T, dtype=float)
        u = np.empty(T, dtype=float)
        tau = np.empty(T, dtype=float)

        log_h[0] = float(np.log(np.var(r[: min(50, max(2, T))])))
        z[0] = r[0] / np.sqrt(np.exp(log_h[0]))
        u[0] = np.log(x[0]) - xi - log_h[0]

        for t in range(1, T):
            tau[t - 1] = tau_1 * z[t - 1] + tau_2 * (z[t - 1] ** 2 - 1.0)
            # NOTE: use delta1 (state coefficient) here — not the AST mixing 'delta'
            log_h[t] = omega + beta * log_h[t - 1] + tau[t - 1] + delta1 * u[t - 1]
            z[t] = r[t] / np.sqrt(np.exp(log_h[t]))
            u[t] = np.log(x[t]) - xi - log_h[t]

        h = np.exp(log_h)
        if np.any(~np.isfinite(h)) or np.any(h <= 0):
            return 1e10

        # AST log-likelihood for returns
        # I_t = 1{ m + s * (r_t / sqrt(h_t)) > 0 }
        z_ast = m + s * (r / np.sqrt(h))
        I = (z_ast > 0.0).astype(float)

        term1 = 0.5 * (v1 + 1.0) * np.log(1.0 + (1.0 / v1) * (z_ast / (2.0 * a_star)) ** 2)
        term2 = 0.5 * (v2 + 1.0) * np.log(1.0 + (1.0 / v2) * (z_ast / (2.0 * (1.0 - a_star))) ** 2)
        ll_r = np.log(s) + np.log(Bv) - 0.5 * np.log(h) - (1.0 - I) * term1 - I * term2

        # Gaussian measurement log-likelihood on log x
        ll_x = -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma_u) + (u ** 2) / (sigma_u ** 2))

        ll = ll_r + ll_x
        if not np.all(np.isfinite(ll)):
            return 1e10

        return float(-np.mean(ll))

    # ---------- IC ----------
    def _ic(self, total_ll: float, k: int) -> Tuple[float, float]:
        T = self.log_returns.size
        aic = 2.0 * k - 2.0 * total_ll
        bic = np.log(T) * k - 2.0 * total_ll
        return float(aic), float(bic)

    # ---------- Estimation ----------
    def optimize(self,initial_params: Optional[Dict[str, float]] = None,compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        MLE via SLSQP with:
          - bounds enforcing sigma_u>0, delta in (0,1), v1>2, v2>2
          - stationarity constraint beta<1
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = dict(
                    omega=float(np.log(np.var(self.log_returns[: min(50, self.log_returns.size)]))),
                    beta=0.55,
                    delta1=0.41,
                    xi=float(np.log(max(self.x.mean(), 1e-6))),
                    tau_1=-0.07,
                    tau_2=0.07,
                    sigma_u=0.38,
                    delta=0.30,  # AST mixing
                    v1=4.0,
                    v2=4.0,
                )

        keys = list(initial_params.keys())
        x0 = [float(initial_params[k]) for k in keys]

        # Bounds in the same order as keys
        bounds = [
            (-np.inf, np.inf),   # omega
            (0.0, 0.999999),     # beta
            (0.0, np.inf),       # delta1
            (-np.inf, np.inf),   # xi
            (-np.inf, np.inf),   # tau_1
            (-np.inf, np.inf),   # tau_2
            (1e-5, np.inf),      # sigma_u
            (1e-6, 1.0 - 1e-6),  # delta (AST)
            (2.00001, np.inf),   # v1
            (2.00001, np.inf),   # v2
        ]

        # Explicit stationarity (redundant with beta upper bound, but clearer):
        def _stab(pars: List[float]) -> float:
            d = dict(zip(keys, pars))
            return 1.0 - d["beta"]

        constraints = ({"type": "ineq", "fun": _stab},)

        res = opt.minimize( self._avg_neg_loglik,x0,method="SLSQP",bounds=bounds,constraints=constraints,options={"maxiter": 500})
        self.convergence = bool(res.success)

        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in res.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")
            return self.optimal_params if self.optimal_params is not None else dict(zip(keys, x0))

        if compute_metrics:
            T = self.log_returns.size
            total_ll = float(-res.fun * T)  # res.fun is avg negative ll
            self.log_likelihood_value = total_ll
            k = len(keys)
            self.aic, self.bic = self._ic(total_ll, k)

            par_vec = np.array([self.optimal_params[k] for k in keys], dtype=float)
            H_avg = approx_hess1(par_vec, self._avg_neg_loglik)  # Hessian of avg neg-ll
            H = H_avg * T
            try:
                cov = inv(H)
                ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                ses = np.full(k, np.nan)

            if np.isnan(ses).any():
                eps = float(np.sqrt(np.finfo(float).eps))
                g = approx_fprime(par_vec, self._avg_neg_loglik, eps)
                try:
                    cov_fb = inv(np.outer(g, g) * T)
                    ses = np.sqrt(np.maximum(np.diag(cov_fb), 0.0))
                except np.linalg.LinAlgError:
                    ses = np.full(k, np.nan)

            self.standard_errors = ses

            return ( self.optimal_params,float(self.aic),float(self.bic),float(self.log_likelihood_value),self.standard_errors)

        return self.optimal_params

    # ---------- Forecasting ----------
    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Forecast h_{T+1}, …, h_{T+h} (variance scale).

        We rebuild in-sample log h_t, then hold the one-step-ahead shocks
        (tau_T, u_T) constant for steps >= 2.
        """
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")

        omega = float(self.optimal_params["omega"])
        beta = float(self.optimal_params["beta"])
        delta1 = float(self.optimal_params["delta1"])  # state coefficient (fixes bug)
        xi = float(self.optimal_params["xi"])
        tau_1 = float(self.optimal_params["tau_1"])
        tau_2 = float(self.optimal_params["tau_2"])

        r = self.log_returns
        x = self.x
        T = r.size

        log_h = np.empty(T, dtype=float)
        z = np.empty(T, dtype=float)
        u = np.empty(T, dtype=float)
        tau = np.empty(T, dtype=float)

        # Rebuild in-sample
        log_h[0] = float(np.log(np.var(r[: min(50, max(2, T))])))
        z[0] = r[0] / np.sqrt(np.exp(log_h[0]))
        u[0] = np.log(x[0]) - xi - log_h[0]

        for t in range(1, T):
            tau[t - 1] = tau_1 * z[t - 1] + tau_2 * (z[t - 1] ** 2 - 1.0)
            log_h[t] = omega + beta * log_h[t - 1] + tau[t - 1] + delta1 * u[t - 1]
            z[t] = r[t] / np.sqrt(np.exp(log_h[t]))
            u[t] = np.log(x[t]) - xi - log_h[t]

        # Freeze last shocks
        tau_last = tau_1 * z[-1] + tau_2 * (z[-1] ** 2 - 1.0)
        u_last = np.log(x[-1]) - xi - log_h[-1]

        forecasts = []
        prev = log_h[-1]
        for _ in range(int(horizon)):
            prev = omega + beta * prev + tau_last + delta1 * u_last
            forecasts.append(np.exp(prev))

        return np.asarray(forecasts, dtype=float)
