import numpy as np
import scipy
from scipy.linalg import inv
from statsmodels.tools.numdiff import approx_hess1
from scipy.optimize import approx_fprime
from scipy.special import gamma


class AR1astModel:
    model_name = "AR-AST"
    distribution = "AST"
    only_kernel = True

    def __init__(self, kernel_array):
        # Initialize instance variables. We use the kernel array
        self.kernel = np.asarray(kernel_array)
        self.optimal_params = None
        self.log_likelihood_value = None
        self.aic = None
        self.bic = None
        self.convergence = None
        self.standard_errors = None

    def K(self, v):
        return gamma((v + 1) / 2) / (np.sqrt(np.pi * v) * gamma(v / 2))

    def B(self, delta, v1, v2):
        return delta * self.K(v1) + (1 - delta) * self.K(v2)

    def alpha_star(self, delta, v1, v2):
        B = self.B(delta, v1, v2)
        return (delta * self.K(v1)) / B

    def m(self, delta, v1, v2):
        a_star = self.alpha_star(delta, v1, v2)
        B = self.B(delta, v1, v2)
        return 4 * B * (-a_star**2 * v1/(v1-1) + (1-a_star)**2 * v2/(v2-1))

    def s(self, delta, v1, v2):
        a_star = self.alpha_star(delta, v1, v2)
        B = self.B(delta, v1, v2)
        m = self.m(delta, v1, v2)
        return np.sqrt(4*(delta*a_star**2*v1/(v1-2) + (1-delta)*(1-a_star)**2*v2/(v2-2)) - m**2)

    def I_t(self, eps, mu_t, h_t, m, s):
        return 1 if (m + s*(eps - mu_t)/np.sqrt(h_t)) > 0 else 0

    def log_likelihood(self, params):
        # params: mu, phi, delta, v1, v2
        mu, phi, delta, v1, v2 = params
        y = self.kernel
        n = len(y)-1
        # residuals with intercept
        eps = y[1:] - mu - phi*(y[:-1] - mu)
        m = self.m(delta, v1, v2)
        s = self.s(delta, v1, v2)
        a_star = self.alpha_star(delta, v1, v2)
        B = self.B(delta, v1, v2)

        ll = np.zeros(n)
        for t in range(n):
            mu_t = mu + phi*(y[t] - mu)
            h_t = 1.0
            I = self.I_t(eps[t], mu, h_t, m, s)
            term1 = (v1+1)/2 * np.log(1 + (1/v1)*((m + s*eps[t])/(2*a_star))**2)
            term2 = (v2+1)/2 * np.log(1 + (1/v2)*((m + s*eps[t])/(2*(1-a_star)))**2)
            ll[t] = np.log(s) + np.log(B) - (1-I)*term1 - I*term2
        # negative average log-lik
        return -np.mean(ll)

    def compute_aic_bic(self, total_loglik, num_params):
        n = len(self.kernel) - 1
        aic = 2*num_params - 2*total_loglik
        bic = np.log(n)*num_params - 2*total_loglik
        return aic, bic

    def optimize(self, initial_params=None, compute_metrics=False):
        if initial_params is None:
            initial_params = {"mu": np.mean(self.kernel), "phi": 0.1, "delta": 0.5, "v1": 5.0, "v2": 5.0}
        keys = list(initial_params.keys())
        x0 = list(initial_params.values())
        n = len(self.kernel) - 1

        bounds = [(-np.inf, np.inf), (-0.999,0.999), (1e-6,1.0), (2.000001,None), (2.000001,None)]
        res = scipy.optimize.minimize(self.log_likelihood, x0, method='SLSQP', bounds=bounds)

        self.convergence = res.success
        if self.convergence:
            self.optimal_params = dict(zip(keys, res.x))

        if compute_metrics and self.convergence:
            tot_ll = -res.fun * n
            self.log_likelihood_value = tot_ll
            k = len(x0)
            self.aic, self.bic = self.compute_aic_bic(tot_ll, k)
            H = approx_hess1(res.x, self.log_likelihood, args=())
            cov = inv(H)/n
            ses = np.sqrt(np.diag(cov))
            if np.isnan(ses).any():
                eps = np.sqrt(np.finfo(float).eps)
                grad = approx_fprime(res.x, self.log_likelihood, eps)
                cov = np.outer(grad, grad)
                ses = np.sqrt(np.diag(cov))
            self.standard_errors = ses
            return self.optimal_params, self.aic, self.bic, self.log_likelihood_value, self.standard_errors
        return self.optimal_params

    def one_step_ahead_forecast(self):
        if self.optimal_params is None:
            raise RuntimeError("Optimize model first.")
        mu = self.optimal_params['mu']
        phi = self.optimal_params['phi']
        return mu + phi*(self.kernel[-1]-mu)

    def multi_step_ahead_forecast(self, horizon):
        if self.optimal_params is None:
            raise RuntimeError("Optimize model first.")
        mu = self.optimal_params['mu']
        phi = self.optimal_params['phi']
        last = self.kernel[-1]
        forecasts = np.empty(horizon)
        forecasts[0] = mu + phi*(last-mu)
        for h in range(1, horizon):
            forecasts[h] = mu + phi*(forecasts[h-1]-mu)
        return forecasts