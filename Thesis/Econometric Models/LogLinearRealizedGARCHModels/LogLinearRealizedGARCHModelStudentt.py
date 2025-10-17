from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1, approx_fprime
from scipy.special import gammaln


class LogLinearRealizedGARCHModelStudentt:
    """
    Log-linear Realized GARCH(1,1) with Student-t returns and Gaussian
    measurement equation on log(x).

    State (log variance):
        log h_t = omega + beta * log h_{t-1} + gamma * log x_{t-1}

    Returns block (Student t, df=nu>2):
        r_t | h_t ~ t_nu(0, scale^2 = h_t * (nu-2)/nu)  [implemented in log form]

    Measurement (Gaussian on log x):
        u_t = log x_t - xi - phi * log h_t - (tau_1 z_t + tau_2 (z_t^2 - 1)) ~ N(0, sigma_u^2)
        where z_t = r_t / sqrt(h_t)
    """

    model_name: str = "LLRGARCH-t"
    distribution: str = "Student t"

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
        self.log_likelihood_value: Optional[float] = None  # total log-likelihood over T
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None
        self.standard_errors: Optional[np.ndarray] = None

    # ---------- Likelihood ----------
    def _avg_neg_loglik(self, par_list: List[float]) -> float:
        """
        Average *negative* log-likelihood to be minimized.

        Parameter order (natural scale):
          [omega, beta, gamma, xi, phi, tau_1, tau_2, sigma_u, nu1]
        Constraints enforced via bounds/guards:
          - sigma_u > 0
          - nu1 > 2
        """
        (omega, beta, gamma, xi, phi, tau_1, tau_2, sigma_u, nu) = [float(v) for v in par_list]

        # Basic parameter checks
        if sigma_u <= 0 or not np.isfinite(sigma_u):
            return 1e10
        if nu <= 2 or not np.isfinite(nu):
            return 1e10
        if not np.all(np.isfinite([omega, beta, gamma, xi, phi, tau_1, tau_2])):
            return 1e10

        r = self.log_returns
        x = self.x
        T = r.size
        if T < 2:
            return 1e10

        # Build log variance
        log_h = np.empty(T, dtype=float)
        log_h[0] = float(np.log(np.var(r[: min(50, max(2, T))])))
        for t in range(1, T):
            log_h[t] = omega + beta * log_h[t - 1] + gamma * np.log(x[t - 1])

        if not np.all(np.isfinite(log_h)):
            return 1e10

        h = np.exp(log_h)
        if np.any(h <= 0) or not np.all(np.isfinite(h)):
            return 1e10

        # Standardized returns & measurement innovation
        z = r / np.sqrt(h)
        tau = tau_1 * z + tau_2 * (z**2 - 1.0)
        u = np.log(x) - xi - phi * log_h - tau

        # Student-t log-likelihood for returns (aligned with user's specification)
        # ll_r = log Γ((ν+1)/2) − log Γ(ν/2) − 0.5 log((ν−2)π) − 0.5 log h − (ν+1)/2 log(1 + r^2 / ((ν−2) h))
        cst = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log((nu - 2.0) * np.pi)
        ll_r = cst - 0.5 * np.log(h) - 0.5 * (nu + 1.0) * np.log(1.0 + (r**2) / ((nu - 2.0) * h))

        # Gaussian measurement log-likelihood on log x
        ll_x = -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma_u) + (u**2) / (sigma_u**2))

        ll = ll_r + ll_x
        if not np.all(np.isfinite(ll)):
            return 1e10

        return float(-np.mean(ll))

    # ---------- Information Criteria ----------
    def _ic(self, total_ll: float, num_params: int) -> Tuple[float, float]:
        T = self.log_returns.size
        aic = 2.0 * num_params - 2.0 * total_ll
        bic = np.log(T) * num_params - 2.0 * total_ll
        return float(aic), float(bic)

    # ---------- Estimation ----------
    def optimize(self,initial_params: Optional[Dict[str, float]] = None,compute_metrics: bool = False) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        MLE via SLSQP with bounds and a stability constraint:
            beta + phi * gamma < 1
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = dict(
                    omega=float(np.log(np.var(self.log_returns[: min(50, self.log_returns.size)]))),
                    beta=0.55,
                    gamma=0.41,
                    xi=float(np.log(max(self.x.mean(), 1e-6))),
                    phi=1.04,
                    tau_1=-0.07,
                    tau_2=0.07,
                    sigma_u=0.38,
                    nu1=5.0,
                )

        keys = list(initial_params.keys())
        x0 = [float(initial_params[k]) for k in keys]

        # Bounds (match parameter order):
        bounds = [
            (-np.inf, np.inf),  # omega
            (0.0, np.inf),      # beta
            (0.0, np.inf),      # gamma
            (-np.inf, np.inf),  # xi
            (0.0, np.inf),      # phi
            (-np.inf, np.inf),  # tau_1
            (-np.inf, np.inf),  # tau_2
            (1e-5, np.inf),     # sigma_u
            (2.00001, np.inf),  # nu1
        ]

        # Stability: 1 - (beta + phi * gamma) >= 0
        def _stab(pars: List[float]) -> float:
            d = dict(zip(keys, pars))
            return 1.0 - (d["beta"] + d["phi"] * d["gamma"])

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

            # SEs from Hessian of avg neg-ll scaled by T
            par_vec = np.array([self.optimal_params[k] for k in keys], dtype=float)
            H_avg = approx_hess1(par_vec, self._avg_neg_loglik)
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
        Forecast h_{T+1}, …, h_{T+h} on the variance scale assuming
        log-linear recursion and x_t held at its last observed value.
        """
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")

        omega = float(self.optimal_params["omega"])
        beta = float(self.optimal_params["beta"])
        gamma = float(self.optimal_params["gamma"])

        r = self.log_returns
        x = self.x
        T = r.size

        # Rebuild in-sample log_h
        log_h = np.empty(T, dtype=float)
        log_h[0] = float(np.log(np.var(r[: min(50, max(2, T))])))
        for t in range(1, T):
            log_h[t] = omega + beta * log_h[t - 1] + gamma * np.log(x[t - 1])

        log_h_last = log_h[-1]
        lx_last = float(np.log(x[-1]))

        # Forward iterate
        fcst = []
        prev = log_h_last
        for _ in range(int(horizon)):
            nxt = omega + beta * prev + gamma * lx_last
            fcst.append(np.exp(nxt))  # back to variance scale
            prev = nxt

        return np.asarray(fcst, dtype=float)
