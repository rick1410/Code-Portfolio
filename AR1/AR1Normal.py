import numpy as np
import scipy
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.optimize import approx_fprime

class AR1Model:
    # Class variables
    model_name = "AR"
    distribution = 'Normal'
    only_kernel = True


    def __init__(self, kernel_array):
        self.log_returns = np.asarray(kernel_array)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def log_likelihood(self, params):
        mu, phi, sigma2 = params
        r = self.log_returns
        residuals = r[1:] - mu - phi * (r[:-1] - mu)
        ll = -0.5 * (np.log(2 * np.pi * sigma2) + (residuals**2) / sigma2)
        return -np.mean(ll)

    def compute_aic_bic(self, total_loglik, num_params):
        n = len(self.log_returns)
        aic = 2 * num_params - 2 * total_loglik
        bic = np.log(n) * num_params - 2 * total_loglik
        return aic, bic

    def optimize(self, initial_params=None, compute_metrics=False):
        if initial_params is None:
            if self.optimal_params is not None:
                initial_params = self.optimal_params
            else:
                mu0 = np.mean(self.log_returns)
                phi0 = 0.1
                sigma20 = np.var(self.log_returns)
                initial_params = {"mu": mu0, "phi": phi0, "sigma2": sigma20}
        keys = list(initial_params.keys())
        x0 = list(initial_params.values())

        bounds = [(None, None), (-0.999, 0.999), (1e-6, None)]
        result = scipy.optimize.minimize(self.log_likelihood,x0,method='SLSQP',bounds=bounds)
        self.convergence = result.success
        if self.convergence:
            self.optimal_params = dict(zip(keys, result.x))
        else:
            print(f"Warning: Optimization failed for {self.model_name}. Retaining previous parameters.")

        if compute_metrics and self.convergence:
            # Recover total log-likelihood
            n = len(self.log_returns)
            self.log_likelihood_value = -result.fun * n
            num_params = len(x0)
            self.aic, self.bic = self.compute_aic_bic(self.log_likelihood_value, num_params)

            # Hessian and standard errors
            H = approx_hess1(list(self.optimal_params.values()), self.log_likelihood, args=())
            cov = inv(H) / n
            ses = np.sqrt(np.diag(cov))
            if np.isnan(ses).any():
                eps = np.sqrt(np.finfo(float).eps)
                grad = approx_fprime(list(self.optimal_params.values()), self.log_likelihood, eps)
                cov_alt = np.outer(grad, grad)
                ses = np.sqrt(np.diag(cov_alt))
            self.standard_errors = ses

            return (self.optimal_params,self.aic,self.bic,self.log_likelihood_value,self.standard_errors)
        return self.optimal_params

    def one_step_ahead_forecast(self):
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")
        mu = self.optimal_params["mu"]
        phi = self.optimal_params["phi"]
        return mu + phi * (self.log_returns[-1] - mu)

    def multi_step_ahead_forecast(self, horizon):
        
        if self.optimal_params is None:
            raise ValueError("Model must be optimized before forecasting.")
        mu = self.optimal_params["mu"]
        phi = self.optimal_params["phi"]
        last = self.log_returns[-1]

        forecasts = np.empty(horizon)
        forecasts[0] = mu + phi * (last - mu)
        for h in range(1, horizon):
            forecasts[h] = mu + phi * (forecasts[h-1] - mu)

        return forecasts