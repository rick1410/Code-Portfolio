from typing import Dict, Optional, Tuple, List
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1

class BayesianGARCHModel:
    """
    Bayesian GARCH(1,1) model with Normal innovations.

    Parameters
    ----------
    log_returns : ArrayLike
        1-D series of returns.

    Attributes
    ----------
    model_name : str
        Short model name ("BGARCH").
    distribution : str
        Innovation distribution ("Normal").
    log_returns : NDArray[np.float_]
        Stored input series as float array.
    bayesian : bool
        Flag indicating Bayesian estimation is used (always True).
    optimal_params : Optional[Dict[str, float]]
        Optimized parameters from likelihood: {"omega","alpha","beta"}.
    chain : Optional[NDArray[np.float_]]
        Posterior draws after burn-in, shape (ndraws - burn_in, 3).
    mcmc_params : Optional[Dict[str, float]]
        Posterior means mapped to parameter names.
    acceptance_rate : Optional[float]
        MH acceptance rate in percent.
    burn_in : int
        Number of initial draws to discard.
    ndraws : int
        Total number of MH iterations (including burn-in).
    confidence_intervals : Optional[List[NDArray[np.float_]]]
        95% equal-tail intervals per parameter: [array([2.5,97.5]), ...].
    convergence : Optional[bool]
        Optimizer success flag.
    candidate_correlations : Optional[List[float]]
        First-order serial correlation of proposal coordinates.
    """

    model_name: str = "BGARCH"
    distribution: str = "Normal"

    log_returns: NDArray[np.float_]
    bayesian: bool
    optimal_params: Optional[Dict[str, float]]
    chain: Optional[NDArray[np.float_]]
    mcmc_params: Optional[Dict[str, float]]
    acceptance_rate: Optional[float]
    burn_in: int
    ndraws: int
    confidence_intervals: Optional[List[NDArray[np.float_]]]
    convergence: Optional[bool]
    candidate_correlations: Optional[List[float]]

    def __init__(self, log_returns: ArrayLike) -> None:
        """Initialize the Bayesian GARCH(1,1) model with a returns series."""
        self.log_returns = np.asarray(log_returns, dtype=float)
        self.bayesian = True
        self.optimal_params = None
        self.chain = None
        self.mcmc_params = None
        self.acceptance_rate = None
        self.burn_in = 1
        self.ndraws = 10
        self.confidence_intervals = None
        self.convergence = None
        self.candidate_correlations = None

    def log_likelihood(self, params: ArrayLike) -> float:
        """
        Compute the (Gaussian) log-likelihood of the GARCH(1,1) model.

        Parameters
        ----------
        params : array-like of shape (3,)
            Parameter vector [omega, alpha, beta].

        Returns
        -------
        float
            Total log-likelihood (sum over t).
        """
        r = self.log_returns
        T = len(r)
        omega, alpha, beta = params

        sig = np.zeros(T, dtype=float)
        sig[0] = np.var(r[:50]) if T > 50 else np.var(r)

        for t in range(0, T - 1):
            sig[t + 1] = omega + alpha * r[t] ** 2 + beta * sig[t]

        log_likelihood = 0.0
        for t in range(T):
            log_likelihood += -(0.5) * np.log(2 * np.pi) - 0.5 * np.log(sig[t]) - 0.5 * (r[t] ** 2) / sig[t]
        return float(log_likelihood)

    def objective(self, params: ArrayLike) -> float:
        """
        Negative log-likelihood objective (for minimization).

        Parameters
        ----------
        params : array-like
            Parameter vector [omega, alpha, beta].

        Returns
        -------
        float
            Negative log-likelihood.
        """
        return -self.log_likelihood(params)

    def serial_correlation(self, data: ArrayLike) -> float:
        """
        Compute lag-1 serial correlation for a vector.

        Parameters
        ----------
        data : array-like
            1-D array.

        Returns
        -------
        float
            Corr(data[:-1], data[1:]).
        """
        x = np.asarray(data, dtype=float)
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    def randomWalkMetropolisHastings(self, initial_params: Dict[str, float], proposal_cov: NDArray[np.float_]) -> Tuple[NDArray[np.float_], float, List[float]]:
        """
        Random-walk Metropolis–Hastings sampler with Gaussian proposals.

        Parameters
        ----------
        initial_params : dict
            Starting values {"omega","alpha","beta"}.
        proposal_cov : ndarray
            Proposal covariance matrix (3x3).

        Returns
        -------
        (ndarray, float, list[float])
            (chain_after_burn_in, acceptance_rate_percent, proposal_coord_serial_corr).
        """
        thetas = np.empty((self.ndraws, 3), dtype=float)
        thetas[0] = np.array(list(initial_params.values()), dtype=float)

        accepted_indices: List[int] = []
        candidate_list: List[NDArray[np.float_]] = []

        current_logpost = self.log_likelihood(thetas[0])

        for i in range(1, self.ndraws):
            candidate = np.random.multivariate_normal(thetas[i - 1], proposal_cov)
            candidate_list.append(candidate)
            omega, alpha, beta = candidate

            if (omega <= 0) or (alpha < 0) or (beta < 0) or ((alpha + beta) >= 1):
                acceptance_prob = 0.0
            else:
                candidate_logpost = self.log_likelihood(candidate)
                ratio = np.exp(candidate_logpost - current_logpost)
                acceptance_prob = float(min(ratio, 1.0))

            u = np.random.rand()
            if u <= acceptance_prob:
                thetas[i] = candidate
                current_logpost = candidate_logpost if acceptance_prob > 0 else current_logpost
                accepted_indices.append(i)
            else:
                thetas[i] = thetas[i - 1]

        self.chain = thetas[self.burn_in:]
        candidate_list = candidate_list[self.burn_in:]
        candidate_array = np.array(candidate_list, dtype=float) if len(candidate_list) > 0 else np.empty((0, 3), dtype=float)

        if candidate_array.size > 0:
            self.candidate_correlations = [self.serial_correlation(candidate_array[:, p]) for p in range(candidate_array.shape[1])]
        else:
            self.candidate_correlations = [np.nan, np.nan, np.nan]

        self.acceptance_rate = float(len(accepted_indices) / self.ndraws * 100.0)
        return self.chain, self.acceptance_rate, self.candidate_correlations

    def optimize(self, initial_params: Optional[Dict[str, float]] = None, compute_metrics: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, float], List[NDArray[np.float_]], float, List[float]]:
        """
        Optimize the likelihood to initialize the MH sampler; optionally run MH and summarize.

        Parameters
        ----------
        initial_params : dict or None
            Initial guess {"omega","alpha","beta"}. If None, uses previous optimum if set,
            else defaults to omega = var(r[:50])*(1-alpha-beta) with alpha =0.1, β=0.7.
        compute_metrics : bool, default False
            If True, returns posterior means, 95% CIs, acceptance rate, and proposal serial correlations.

        Returns
        -------
        dict or tuple
            If `compute_metrics` is False:
                optimal_params
            If `compute_metrics` is True and converged:
                (optimal_params, mcmc_params, confidence_intervals, acceptance_rate, candidate_correlations)
        """
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                v0 = np.var(self.log_returns[:50]) if self.log_returns.size > 50 else np.var(self.log_returns)
                initial_params = {"omega": float(v0 * (1 - 0.1 - 0.7)), "alpha": 0.1, "beta": 0.7}

        keys = list(initial_params.keys())
        x0 = np.array(list(initial_params.values()), dtype=float)

        bounds = [(1e-5, 100.0), (0.0, 10.0), (0.0, 10.0)]
        constraints = {'type': 'ineq', 'fun': lambda x: 1 - x[1] - x[2]}

        result = scipy.optimize.minimize(self.objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        self.convergence = bool(result.success)
        if self.convergence:
            print(f"Model: {self.model_name} | Convergence: Success")
            self.optimal_params = dict(zip(keys, result.x))

            hessian = approx_hess1(list(self.optimal_params.values()), self.objective, args=())
            hessian_inv = inv(hessian)

            self.chain, self.acceptance_rate, self.candidate_correlations = self.randomWalkMetropolisHastings(self.optimal_params, hessian_inv)
            mean_params = np.mean(self.chain, axis=0)
            self.mcmc_params = dict(zip(keys, mean_params))
            self.confidence_intervals = [np.percentile(self.chain[:, i], [2.5, 97.5]) for i in range(3)]

        if self.convergence and compute_metrics:
            return self.optimal_params, self.mcmc_params, self.confidence_intervals, self.acceptance_rate, self.candidate_correlations
        else:
            return self.optimal_params

    def one_step_ahead_forecast(self) -> float:
        """
        Compute one-step-ahead forecast of sigma_{T+1}^2 using posterior mean parameters.

        Returns
        -------
        float
            Forecast variance for time T+1.

        Raises
        ------
        ValueError
            If the model has not been fitted (no `mcmc_params`).
        """
        if self.mcmc_params is None:
            raise ValueError("Fit model before forecasting.")
        omega = self.mcmc_params['omega']
        alpha = self.mcmc_params['alpha']
        beta = self.mcmc_params['beta']
        r = self.log_returns
        T = len(r)
        sigma2 = np.empty(T, dtype=float)
        sigma2[0] = np.var(r[:50]) if T > 50 else np.var(r)
        for t in range(1, T):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        return float(omega + alpha * r[-1] ** 2 + beta * sigma2[-1])

    def multi_step_ahead_forecast(self, horizon: int) -> NDArray[np.float_]:
        """
        Compute h-step-ahead variance forecasts using posterior mean parameters.

        Parameters
        ----------
        horizon : int
            Number of steps ahead (h >= 1).

        Returns
        -------
        np.ndarray
            Array of length `horizon` with forecasts for sigma_{T+1}^2, …, sigma_{T+h}^2.

        Raises
        ------
        ValueError
            If the model has not been fitted (no `mcmc_params`).
        """
        if self.mcmc_params is None:
            raise ValueError("Fit model before forecasting.")
        omega = self.mcmc_params['omega']
        alpha = self.mcmc_params['alpha']
        beta = self.mcmc_params['beta']
        r = self.log_returns
        T = len(r)
        sigma2 = np.empty(T, dtype=float)
        sigma2[0] = np.var(r[:50]) if T > 50 else np.var(r)
        for t in range(1, T):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        forecasts = np.empty(horizon, dtype=float)
        forecasts[0] = omega + alpha * r[-1] ** 2 + beta * sigma2[-1]
        for h in range(1, horizon):
            forecasts[h] = omega + (alpha + beta) * forecasts[h - 1]
        return forecasts
