from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats  # kept for compatibility with original imports
from scipy.optimize import minimize
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from DataAndRealizedKernel.PDFmodels import PDFModels


class BootstrapEB2FilterSV:
    """
    Bootstrap particle filter for a stochastic volatility model with an EGB2 return density.

    The observation used for the state filter is the log-squared demeaned return:
        x_t = log((r_t - \\bar{r})^2).
    The latent log-variance state follows:
        alpha_{t+1} = psi + phi * alpha_t + eta_t,   eta_t ~ N(0, var_eta).

    A Kalman filter is used for the Gaussian state-space on `x_t`, while a simple bootstrap
    particle filter supplies filtered states using EGB2 measurement weights for returns.

    Attributes
    ----------
    model_name : str
        Short model identifier.
    distribution : str
        Innovation distribution label (kept as in original code).
    pdf_model : PDFModels
        Helper object that provides EGB2 pdf.
    log_returns : np.ndarray
        Input log returns r_t.
    mean_return : float
        Sample mean of `log_returns`.
    demeaned_returns : np.ndarray
        r_t - mean_return.
    x_t : np.ndarray
        log((r_t - mean_return)^2), observation for the state filter.
    optimal_params : Optional[Dict[str, float]]
        Optimized parameters if available, keys: {'psi','phi','var_eta'}.
    log_likelihood_value : Optional[float]
        Total log-likelihood at optimum (not averaged).
    aic : Optional[float]
        Akaike Information Criterion (if computed).
    bic : Optional[float]
        Bayesian Information Criterion (if computed).
    convergence : Optional[bool]
        Optimization convergence flag.
    standard_errors : Optional[np.ndarray]
        Asymptotic standard errors from inverse Hessian (if computed).
    dD : float
        Constant offset in the measurement equation (kept as in original).
    """

    model_name: str = "BFSV-EGB2"
    distribution: str = "Normal"

    def __init__(self, log_returns: np.ndarray) -> None:
        """
        Initialize containers and precompute demeaned series and log-squared returns.

        Parameters
        ----------
        log_returns : np.ndarray
            Array of log returns r_t.
        """
        self.pdf_model: PDFModels = PDFModels()
        self.log_returns: np.ndarray = np.asarray(log_returns, dtype=float).ravel()
        self.mean_return: float = float(np.mean(self.log_returns))
        self.demeaned_returns: np.ndarray = self.log_returns - self.mean_return
        self.x_t: np.ndarray = np.log(self.demeaned_returns**2)
        self.optimal_params: Optional[Dict[str, float]] = None
        self.log_likelihood_value: Optional[float] = None
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.convergence: Optional[bool] = None
        self.standard_errors: Optional[np.ndarray] = None
        self.dD: float = -1.27

    def kalman_filter(self, x: np.ndarray, psi: float, phi: float, var_eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a univariate Kalman filter with measurement `x_t = alpha_t + dD + eps_t`, eps_t ~ N(0, pi^2/2).

        Parameters
        ----------
        x : np.ndarray
            Observations (log-squared demeaned returns), shape (T,).
        psi : float
            Intercept in the state equation.
        phi : float
            AR(1) parameter in the state equation.
        var_eta : float
            State innovation variance.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (v, F, alpha, P, alpha_filt) with prediction errors, variances,
            predicted states/variances, and filtered state means.
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
        Gaussian prediction-error negative log-likelihood from the Kalman filter.

        Parameters
        ----------
        x : np.ndarray
            Observations (log-squared demeaned returns).
        psi : float
            State intercept.
        phi : float
            State AR(1) parameter.
        var_eta : float
            State innovation variance.

        Returns
        -------
        float
            Negative log-likelihood value (to be minimized).
        """
        v, F, _, _, _ = self.kalman_filter(x, psi, phi, var_eta)
        return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(F) + (v**2) / F))

    def objective(self, params: List[float], x: np.ndarray) -> float:
        """
        Wrapper objective enforcing parameter domain before evaluating likelihood.

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
        if 0 < phi < 0.9999999999 and var_eta > 0:  # single-line if permitted
            return self.log_likelihood(x, psi, phi, var_eta)
        else:
            return 1e10

    def moment_estimators(self) -> List[float]:
        """
        Crude moment-based initial values for (psi, phi, var_eta).

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
        Estimate (psi, phi, var_eta) via SLSQP under bounds, optionally compute AIC/BIC/SEs.

        Parameters
        ----------
        initial_params : Optional[Dict[str, float]], default None
            If None, reuses previous optimum if present; else uses moment-based initials.
        compute_metrics : bool, default True
            If True, compute AIC, BIC, total log-likelihood, and standard errors.

        Returns
        -------
        Union[Dict[str, float], Tuple[Dict[str, float], float, float, float, np.ndarray]]
            If `compute_metrics` is True, returns (params, aic, bic, loglik, std_errs); else params only.
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

    def conditional_probability(self, y: float, H: float, xi: float) -> float:
        """
        One-step conditional probability under EGB2 for a given return and state.

        Parameters
        ----------
        y : float
            Demeaned return at time t.
        H : float
            Latent state at time t.
        xi : float
            Long-run component psi / (1 - phi).

        Returns
        -------
        float
            Conditional density value under standardized EGB2(0,1,p=2,q=2).
        """
        sigma = float(np.exp((xi + H) / 2.0))
        y_standardized = y / sigma
        conditional_prob = PDFModels().egb2_pdf(y_standardized, 0.0, 1.0, 2.0, 2.0)
        return float(conditional_prob)

    def bootstrap(self) -> np.ndarray:
        """
        Bootstrap particle filter to obtain filtered states using EGB2 measurement weights.

        Returns
        -------
        np.ndarray
            Filtered state means across time (length T).
        """
        optimal_omega, optimal_phi, optimal_var_eta = list(self.optimal_params.values())
        xi = optimal_omega / (1.0 - optimal_phi)
        uncond_var = optimal_var_eta / (1.0 - optimal_phi**2)
        M = 10
        T = len(self.log_returns)
        states = np.zeros((M, T))
        weights = np.zeros(M)
        result = np.zeros(T)

        for t in range(T):
            for m in range(M):
                if t == 0:
                    states[m, t] = np.random.normal(0.0, np.sqrt(uncond_var))
                    weights[m] = self.conditional_probability(float(self.demeaned_returns[t]), float(states[m, t]), float(xi))
                else:
                    states[m, t] = optimal_phi * states[m, t - 1] + np.random.normal(0.0, np.sqrt(optimal_var_eta))
                    weights[m] = self.conditional_probability(float(self.demeaned_returns[t]), float(states[m, t]), float(xi))
            weights /= np.sum(weights)
            result[t] = np.sum(weights * states[:, t])
            cdf = np.cumsum(weights)
            index = np.searchsorted(cdf, np.random.uniform(size=M))
            states[:, t] = states[:, t][index]

        return result

    def multi_step_ahead_forecast(self, horizon: int) -> np.ndarray:
        """
        Forecast future variances :math:`\\sigma^2_{T+1},\\ldots,\\sigma^2_{T+H}` from the filtered state.

        Uses the last filtered state from `bootstrap()` as the starting point, then iterates
        the noiseless state equation:
            state_{t+1} = phi * state_t,
        mapping to variance via :math:`\\exp(state + \\kappa)` with :math:`\\kappa = \\psi/(1-\\phi)`.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecasted variances.
        """
        if self.optimal_params is None:
            raise RuntimeError("Model must be optimized before forecasting.")

        last_filtered_state = float(self.bootstrap()[-1])
        psi = float(self.optimal_params["psi"])
        phi = float(self.optimal_params["phi"])
        var_eta = float(self.optimal_params["var_eta"])  # kept for parity with original interface
        _ = var_eta  # acknowledged but not used in deterministic propagation
        kappa = psi / (1.0 - phi)

        forecasts: List[float] = []
        previous_state = last_filtered_state
        for _ in range(horizon):
            next_state = phi * previous_state
            sigma2_forecast = float(np.exp(next_state + kappa))
            forecasts.append(sigma2_forecast)
            previous_state = next_state

        return np.asarray(forecasts, dtype=float)
