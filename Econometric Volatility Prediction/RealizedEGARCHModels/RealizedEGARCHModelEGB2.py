from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1, approx_fprime
from scipy.special import gamma, digamma, polygamma


class RealizedEGARCHModelEGB2:
    """
    Realized EGARCH(1,1) with EGB2 returns and Gaussian measurement on log(x).

    State (log variance):
        log h_t = omega + beta * log h_{t-1} + tau_{t-1} + delta * u_{t-1}
        where tau_t = tau_1 * z_t + tau_2 * (z_t^2 - 1),
              z_t = r_t / sqrt(h_t),
              u_t = log x_t - xi - log h_t.

    Returns block (EGB2):
        With parameters p>0, q>0, and
          Δ  = ψ(p) - ψ(q)
          Ω  = ψ₁(p) + ψ₁(q)
        the log-density (up to constants) for r_t | h_t is
          0.5*log Ω + p*(sqrt(Ω)*(r_t)/sqrt(h_t) + Δ) - 0.5*log h_t
          - log(Γ(p)Γ(q)/Γ(p+q)) - (p+q)*log(1 + exp(sqrt(Ω)*(r_t)/sqrt(h_t) + Δ)).

    Measurement (Gaussian on log x):
        u_t ~ N(0, σ_u²).
    """

    model_name: str = "REGARCH-EGB2"
    distribution: str = "EGB2"

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

    # ---------- EGB2 helpers ----------
    @staticmethod
    def _Delta(p: float, q: float) -> float:
        return float(digamma(p) - digamma(q))

    @staticmethod
    def _Omega(p: float, q: float) -> float:
        return float(polygamma(1, p) + polygamma(1, q))

    # ---------- likelihood ----------
    def _avg_neg_loglik(self, par_list: List[float]) -> float:
        """
        Average *negative* log-likelihood to minimize.

        Parameter order (natural scale):
          [omega, beta, delta, xi, tau_1, tau_2, sigma_u, p, q]
          with constraints: sigma_u>0, p>0, q>0, beta<1 (enforced via constraint).
        """
        omega, beta, delt, xi, tau_1, tau_2, sigma_u, p, q = [float(v) for v in par_list]

        # Basic checks
        if sigma_u <= 0 or not np.isfinite(sigma_u):
            return 1e10
        if p <= 0 or q <= 0:
            return 1e10
        if not np.all(np.isfinite([omega, beta, delt, xi, tau_1, tau_2, p, q])):
            return 1e10

        r = self.log_returns
        x = self.x
        T = r.size
        if T < 2:
            return 1e10

        Delta = self._Delta(p, q)
        Omega = self._Omega(p, q)

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
            log_h[t] = omega + beta * log_h[t - 1] + tau[t - 1] + delt * u[t - 1]
            z[t] = r[t] / np.sqrt(np.exp(log_h[t]))
            u[t] = np.log(x[t]) - xi - log_h[t]

        h = np.exp(log_h)
        if np.any(~np.isfinite(h)) or np.any(h <= 0):
            return 1e10

        # EGB2 log-likelihood for returns
        kern = np.sqrt(Omega) * (r / np.sqrt(h)) + Delta
        ll_r = (0.5 * np.log(Omega)+ p * kern - 0.5 * np.log(h) - np.log(gamma(p) * gamma(q) / gamma(p + q)) - (p + q) * np.log1p(np.exp(kern)))

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
          - bounds enforcing sigma_u>0, p>0, q>0
          - explicit stationarity constraint beta<1
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = dict(
                    omega=float(np.log(np.var(self.log_returns[: min(50, self.log_returns.size)]))),
                    beta=0.55,
                    delta=0.41,
                    xi=float(np.log(max(self.x.mean(), 1e-6))),
                    tau_1=-0.07,
                    tau_2=0.07,
                    sigma_u=0.38,
                    p=3.5,
                    q=3.5,
                )

        keys = list(initial_params.keys())
        x0 = [float(initial_params[k]) for k in keys]

        bounds = [
            (-np.inf, np.inf),  # omega
            (0.0, 0.999999),    # beta
            (0.0, np.inf),      # delta (state coefficient)
            (-np.inf, np.inf),  # xi
            (-np.inf, np.inf),  # tau_1
            (-np.inf, np.inf),  # tau_2
            (1e-5, np.inf),     # sigma_u
            (1e-6, np.inf),     # p
            (1e-6, np.inf),     # q
        ]

        def _stab(pars: List[float]) -> float:
            d = dict(zip(keys, pars))
            return 1.0 - d["beta"]

        constraints = ({"type": "ineq", "fun": _stab},)

        res = opt.minimize(
            self._avg_neg_loglik,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )
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

            return (self.optimal_params,float(self.aic),float(self.bic),float(self.log_likelihood_value),self.standard_errors)

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
        delt = float(self.optimal_params["delta"])
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
            log_h[t] = omega + beta * log_h[t - 1] + tau[t - 1] + delt * u[t - 1]
            z[t] = r[t] / np.sqrt(np.exp(log_h[t]))
            u[t] = np.log(x[t]) - xi - log_h[t]

        # Freeze last shocks
        tau_last = tau_1 * z[-1] + tau_2 * (z[-1] ** 2 - 1.0)
        u_last = np.log(x[-1]) - xi - log_h[-1]

        forecasts = []
        prev = log_h[-1]
        for _ in range(int(horizon)):
            prev = omega + beta * prev + tau_last + delt * u_last
            forecasts.append(np.exp(prev))

        return np.asarray(forecasts, dtype=float)
