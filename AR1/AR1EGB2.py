import numpy as np
import scipy.optimize
from scipy.special import gamma, digamma, polygamma
from numpy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.optimize import approx_fprime
class AR1EGB2Model:
    model_name = "AR1-EGB2"
    distribution = "EGB2"

    def __init__(self, kernel_array):
        self.log_returns = kernel_array
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def _egb2_constants(self, p, q):
        Delta = digamma(p) - digamma(q)
        Omega = polygamma(1, p) + polygamma(1, q)
        norm_const = gamma(p) * gamma(q) / gamma(p + q)
        return Delta, Omega, norm_const

    def log_likelihood(self, params):
        phi, p, q = params
        r = self.log_returns
        eps = r[1:] - phi * r[:-1]
        N = eps.size

        Delta, Omega, norm_const = self._egb2_constants(p, q)
        root_Omega = np.sqrt(Omega)

        ll = np.zeros(N)
        for i, e in enumerate(eps):
            z = root_Omega * e + Delta
            ll[i] = (0.5 * np.log(Omega) + p * z - np.log(norm_const) - (p + q) * np.log1p(np.exp(z)))

        # Return negative average
        return -np.mean(ll)

    def compute_aic_bic(self, total_loglik, num_params):
        N = len(self.log_returns) - 1
        aic = 2 * num_params - 2 * total_loglik
        bic = np.log(N) * num_params - 2 * total_loglik
        return aic, bic

    def optimize(self, initial_params=None, compute_metrics=False):
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                initial_params = {"phi": 0.1, "p": 3.5, "q": 3.5}

        keys = list(initial_params.keys())
        x0 = list(initial_params.values())
        N = len(self.log_returns) - 1

        bounds = [(-0.999, 0.999), (2.000001, None), (2.000001, None)]
        result = scipy.optimize.minimize(self.log_likelihood, x0,method='SLSQP', bounds=bounds)

        self.convergence = result.success
        if self.convergence:
            self.optimal_params = dict(zip(keys, result.x))

        if compute_metrics and self.convergence:
            # Total log-likelihood from average returned
            total_ll = -result.fun * N
            self.log_likelihood_value = total_ll
            k = len(x0)
            self.aic, self.bic = self.compute_aic_bic(total_ll, k)

            # Hessian and standard errors
            H = approx_hess1(result.x, self.log_likelihood, args=())
            cov = inv(H) / N
            ses = np.sqrt(np.diag(cov))
            if np.isnan(ses).any():
                eps = np.sqrt(np.finfo(float).eps)
                grad = scipy.optimize.approx_fprime(result.x, self.log_likelihood, eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.diag(cov_alt))
            self.standard_errors = ses

            return (self.optimal_params,self.aic,self.bic,self.log_likelihood_value,self.standard_errors)

        return self.optimal_params

    def one_step_ahead_forecast(self):
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")
        phi = self.optimal_params.get("phi")
        return phi * self.log_returns[-1]

    def multi_step_ahead_forecast(self, horizon):
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")
        phi = self.optimal_params.get("phi")
        last = self.log_returns[-1]
        forecasts = np.empty(horizon)
        forecasts[0] = phi * last
        for h in range(1, horizon):
            forecasts[h] = phi * forecasts[h-1]
        return forecasts


