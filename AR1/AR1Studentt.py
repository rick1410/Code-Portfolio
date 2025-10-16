import numpy as np
import scipy
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.optimize import approx_fprime
from scipy.special import gammaln

class AR1ModelStudentt:
    model_name = "AR-t"
    distribution = "Student t"
    only_kernel = True

    def __init__(self, kernel_array):
        self.log_returns = np.asarray(kernel_array)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = False
        self.standard_errors = None

    def log_likelihood(self, mu, phi, sigma2, nu):
        r = self.log_returns
        eps = (r[1:] - mu) - phi * (r[:-1] - mu)
        ll_i = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * sigma2) - ((nu + 1) / 2) * np.log1p((eps**2) / (nu * sigma2)))
        return -np.mean(ll_i)

    def compute_aic_bic(self, total_loglik, num_params):
        n = len(self.log_returns) - 1
        aic = 2 * num_params - 2 * total_loglik
        bic = np.log(n) * num_params - 2 * total_loglik
        return aic, bic

    def optimize(self, initial_params=None, compute_metrics=False):
        if initial_params is None:
            initial_params = {"mu":np.mean(self.log_returns),"phi": 0.1,"sigma2": np.var(self.log_returns), "nu1": 5.0}
        
        mu0 = initial_params["mu"]
        phi0 = initial_params["phi"]
        sigma20 = initial_params["sigma2"]
        nu0 = initial_params["nu1"]

        def objective(x):
            return self.log_likelihood(x[0], x[1], x[2], x[3])

        x0 = np.array([mu0, phi0, sigma20, nu0])
        bounds = [(None, None),(-0.999, 0.999),(1e-6, None),(2.000001, None)]

        res = scipy.optimize.minimize(objective,x0,method="SLSQP",bounds=bounds)

        self.convergence = res.success
        mu_opt, phi_opt, sigma2_opt, nu_opt = res.x
        self.optimal_params = {"mu": mu_opt,"phi": phi_opt,"sigma2": sigma2_opt,"nu1": nu_opt}

        if compute_metrics and self.convergence:
            n = len(self.log_returns) - 1
            total_ll = -res.fun * n
            self.log_likelihood_value = total_ll
            k = len(x0)
            self.aic, self.bic = self.compute_aic_bic(total_ll, k)

            H = approx_hess1(res.x, objective, args=())
            cov = inv(H) / n
            ses = np.sqrt(np.diag(cov))
            if np.isnan(ses).any():
                eps = np.sqrt(np.finfo(float).eps)
                grad = approx_fprime(res.x, objective, eps)
                cov = np.outer(grad, grad)
                ses = np.sqrt(np.diag(cov))
            self.standard_errors = ses

            return (self.optimal_params,self.aic,self.bic,self.log_likelihood_value,self.standard_errors)

        return self.optimal_params

    def one_step_ahead_forecast(self):
        if self.optimal_params is None:
            raise RuntimeError("Model must be optimized before forecasting.")
        mu  = self.optimal_params["mu"]
        phi = self.optimal_params["phi"]
        last = self.log_returns[-1]
        return mu + phi * (last - mu)

    def multi_step_ahead_forecast(self, horizon):
        if self.optimal_params is None:
            raise RuntimeError("Model must be optimized before forecasting.")
        mu  = self.optimal_params["mu"]
        phi = self.optimal_params["phi"]

        forecasts = np.empty(horizon)
        forecasts[0] = mu + phi * (self.log_returns[-1] - mu)
        for h in range(1, horizon):
            forecasts[h] = mu + phi * (forecasts[h-1] - mu)
        return forecasts