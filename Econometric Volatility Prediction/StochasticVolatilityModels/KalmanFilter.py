from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1


class KalmanFilterSV:
    """
    Kalman filter stochastic volatility (SV) model for log-squared demeaned returns.

    The observation equation uses
        x_t = log((r_t - mean(r))^2) = alpha_t + dD + e_t,     e_t ~ N(0, pi^2/2),
    and the latent state follows an AR(1)
        alpha_{t+1} = psi + phi * alpha_t + eta_t,             eta_t ~ N(0, var_eta).

    Attributes
    ----------
    model_name : str
        Short model identifier.
    distribution : str
        Innovation distribution label.
    log_returns : np.ndarray
        Input return series r_t, shape (T,).
    mean_return : float
        Sample mean of returns.
    x_t : np.ndarray
        log((r_t - mean_return)^2), shape (T,).
    optimal_params : Optional[Dict[str, float]]
        Optimized parameters with keys {'psi','phi','var_eta'}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (not averaged).
    aic : Optional[float]
        Akaike Information Criterion (if computed).
    bic : Optional[float]
        Bayesian Information Criterion (if computed).
    convergence : Optional[bool]
        Optimizer convergence flag.
    standard_errors : Optional[np.ndarray]
        Asymptotic standard errors from inverse Hessian (if computed).
    dD : float
        Constant offset in the measurement equation.
    """

    model_name: str = "KFSV"
    distribution: str = "Normal"

    def __init__(self, log_returns: np.ndarray) -> None:
        """
        Initialize the model and precompute transformed series.

        Parameters
        ----------
        log_returns : np.ndarray
            Array of log returns r_t.
        """
        self.log_returns: np.ndarray = np.asarray(log_returns, dtype=float).ravel()
        self.mean_return: float = float(np.mean(self.log_returns))
        self.x_t: np.ndarray = np.log((self.log_returns - self.mean_return) ** 2)
        self.optimal_params: Optional[Dict[str, float]] = None
        self.log_likelihood_value: Optional[float] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None
        self.standard_errors: Optional[np.ndarray] = None
        self.dD: float = -1.27

    def kalman_filter(self, x: np.ndarray, psi: float, phi: float, var_eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a univariate Kalman filter for the SV measurement equation.

        Parameters
        ----------
        x : np.ndarray
            Observations x_t = log((r_t - mean)^2), shape (T,).
        psi : float
            State intercept.
        phi : float
            AR(1) coefficient for the latent state.
        var_eta : float
            State innovation variance.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Prediction errors v, their variances F, predicted states alpha and P,
            and filtered state means alpha_filt.
        """
        T = len(x)
        v = np.zeros(T)
        F = np.zeros(T)
        alpha = np.zeros(T + 1)
        P = np.zeros(T + 1)
        alpha_filt = np.zeros(T)
        P_filt = np.zeros(T)

        alpha[0] = psi / (1.0 - phi)
        P[0] = var_eta / (1.0 - phi**2)
        obs_var = np.pi**2 / 2.0

        for t in range(T):
            v[t] = x[t] - alpha[t] - self.dD
            F[t] = P[t] + obs_var
            alpha_filt[t] = alpha[t] + (P[t] / F[t]) * v[t]
            P_filt[t] = P[t] - (P[t] ** 2) / F[t]
            if t < T - 1:
                alpha[t + 1] = phi * alpha_filt[t] + psi
                P[t + 1] = phi**2 * P_filt[t] + var_eta

        return v, F, alpha, P, alpha_filt

    def log_likelihood(self, x: np.ndarray, psi: float, phi: float, var_eta: float) -> float:
        """
        Gaussian prediction-error negative log-likelihood (to minimize).

        Parameters
        ----------
        x : np.ndarray
            Observations, shape (T,).
        psi : float
            State intercept.
        phi : float
            AR(1) coefficient.
        var_eta : float
            State innovation variance.

        Returns
        -------
        float
            Negative log-likelihood value.
        """
        v, F, _, _, _ = self.kalman_filter(x, psi, phi, var_eta)
        return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(F) + (v**2) / F))

    def objective(self, params: List[float], x: np.ndarray) -> float:
        """
        Objective wrapper with domain checks for SLSQP.

        Parameters
        ----------
        params : List[float]
            [psi, phi, var_eta].
        x : np.ndarray
            Observations.

        Returns
        -------
        float
            Negative log-likelihood or a large penalty if constraints fail.
        """
        psi, phi, var_eta = params
        if 0 < phi < 0.9999999999 and var_eta > 0: return self.log_likelihood(x, psi, phi, var_eta)
        return 1e10

    def moment_estimators(self) -> List[float]:
        """
        Moment-based initial values for (psi, phi, var_eta).

        Returns
        -------
        List[float]
            [initial_psi, initial_phi, initial_var_eta].
        """
        x = self.x_t
        initial_phi = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        initial_psi = (1.0 - initial_phi) * (float(np.mean(x)) + 1.27)
        initial_var_eta = (1.0 - initial_phi**2) * (float(np.var(x)) - (np.pi**2) / 2.0)
        return [initial_psi, initial_phi, initial_var_eta]

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = True) -> Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]:
        """
        Estimate (psi, phi, var_eta) via SLSQP with bounds; optionally compute AIC/BIC/SEs.

        Parameters
        ----------
        initial_params : Optional[Dict[str, float]], default None
            If None, reuse last optimum when available; else use moment-based initials.
        compute_metrics : bool, default True
            If True, return (params, AIC, BIC, loglik, std_errs); otherwise return params.

        Returns
        -------
        Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]
            Parameter dictionary, or tuple including information criteria and standard errors.
        """
        x = self.x_t
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                psi0, phi0, ve0 = self.moment_estimators()
                initial_params = {"psi": psi0, "phi": phi0, "var_eta": ve0}

        keys: List[str] = list(initial_params.keys())
        init_vals: List[float] = list(initial_params.values())
        bounds = [(-np.inf, np.inf), (1e-3, 0.9999999), (1e-10, np.inf)]

        result = minimize(fun=lambda p: self.objective(p, x), x0=init_vals, method="SLSQP", bounds=bounds)
        self.convergence = bool(result.success)
        if self.convergence:
            self.optimal_params = dict(zip(keys, [float(v) for v in result.x]))
            print(f"Model: {self.model_name} | Convergence: Success")
        else:
            if self.optimal_params is None:
                self.optimal_params = dict(zip(keys, [float(v) for v in result.x]))
                print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")

        if compute_metrics and self.convergence:
            self.log_likelihood_value = -float(result.fun)
            num_params = 3
            T = len(x)
            logL = float(self.log_likelihood_value)
            self.aic = 2 * num_params - 2 * logL
            self.bic = np.log(T) * num_params - 2 * logL
            hessian = approx_hess1(np.asarray(result.x, dtype=float), self.objective, args=(x,))
            hessian_inv = inv(hessian) / max(T, 1)
            self.standard_errors = np.sqrt(np.maximum(np.diag(hessian_inv), 0.0))
            return self.optimal_params, float(self.aic), float(self.bic), float(self.log_likelihood_value), self.standard_errors
        else:
            return self.optimal_params

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Forecast future variances from the last filtered state (deterministic propagation).

        Uses the final filtered state from `kalman_filter` as the starting point, then iterates
        the noiseless state equation:
            state_{t+1} = phi * state_t + psi,
        mapping to variance via :math:`exp(state)`.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecasted variances [variance_{T+1}, …, variance_{T+horizon}].
        """
        if self.optimal_params is None:
            raise RuntimeError("Model must be optimized before forecasting.")

        psi = float(self.optimal_params["psi"])
        phi = float(self.optimal_params["phi"])
        var_eta = float(self.optimal_params["var_eta"])  # acknowledged but unused in deterministic path
        _ = var_eta
        _, _, _, _, alpha_filtered = self.kalman_filter(self.x_t, psi, phi, float(self.optimal_params["var_eta"]))
        last_filtered_state = float(alpha_filtered[-1])

        variance_forecasts: List[float] = []
        state_prev = last_filtered_state
        for _ in range(horizon):
            state_next = phi * state_prev + psi
            variance_forecasts.append(float(np.exp(state_next)))
            state_prev = state_next

        return np.asarray(variance_forecasts, dtype=float)
