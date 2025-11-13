"""
Bayesian AR(1) vs AR(2) model comparison and prediction.

This module defines the `BayesianARModelSelector` class, which:
- Loads a univariate time series (e.g. US real GDP growth).
- Constructs lagged variables and AR(1)/AR(2) design matrices.
- Fits AR(1) and AR(2) via OLS and computes one-step-ahead predictions.
- Performs Bayesian predictive simulation for each model.
- Uses the Savage–Dickey density ratio (SDDR) to obtain a Bayes factor
  and posterior model probabilities.
- Combines AR(1) and AR(2) predictive distributions via Bayesian
  model averaging.

Dependencies
------------
- numpy
- pandas
- matplotlib
- seaborn
- scipy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t


@dataclass
class BayesianARModelSelector:
    """
    Bayesian model comparison between AR(1) and AR(2) with SDDR.

    Parameters
    ----------
    y : np.ndarray
        Full time series y_t (length N).
    time : np.ndarray, optional
        Time index aligned with y (length N). Optional, used only for plotting.
    random_state : int, optional
        Seed for NumPy's random number generator for reproducibility.

    Notes
    -----
    Internally, the model uses the aligned sample t = 3, ..., N for estimation:
        y_t     = y[2:]
        y_{t-1} = y[1:-1]
        y_{t-2} = y[:-2]
    so the effective sample size is T = N - 2.
    """

    y: np.ndarray
    time: Optional[np.ndarray] = None
    random_state: Optional[int] = None

    # Internal fields set after initialization
    y_eff: np.ndarray = field(init=False, repr=False)
    y_lag1: np.ndarray = field(init=False, repr=False)
    y_lag2: np.ndarray = field(init=False, repr=False)
    time_eff: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    X_ar1: np.ndarray = field(init=False, repr=False)
    X_ar2: np.ndarray = field(init=False, repr=False)
    df_ar1: int = field(init=False)
    df_ar2: int = field(init=False)

    # OLS results
    beta_ar1: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    beta_ar2: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    vcov_ar1: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    vcov_ar2: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    sigma2_ar1: Optional[float] = field(init=False, default=None)
    sigma2_ar2: Optional[float] = field(init=False, default=None)

    # Prior for AR(2)
    beta_prior_ar2: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    vcov_prior_ar2: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    df_prior_ar2: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        y = np.asarray(self.y).astype(float)
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array.")

        n = len(y)
        if n < 3:
            raise ValueError("y must have at least 3 observations to form AR(2).")

        # Effective sample (aligned lags)
        self.y_eff = y[2:]
        self.y_lag1 = y[1:-1]
        self.y_lag2 = y[:-2]

        if self.time is not None:
            time_arr = np.asarray(self.time)
            if len(time_arr) != n:
                raise ValueError("time must have the same length as y.")
            self.time_eff = time_arr[2:]
        else:
            self.time_eff = None

        T = len(self.y_eff)

        # Design matrices
        self.X_ar1 = np.column_stack([np.ones(T), self.y_lag1])
        self.X_ar2 = np.column_stack([np.ones(T), self.y_lag1, self.y_lag2])

        # Degrees of freedom (T - number of parameters)
        self.df_ar1 = T - self.X_ar1.shape[1]
        self.df_ar2 = T - self.X_ar2.shape[1]

    # ------------------------------------------------------------------
    # Class constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_excel(cls,file_path: str,y_column: str = "y_t",time_column: Optional[str] = "time",random_state: Optional[int] = None) -> BayesianARModelSelector:
        
        """
        Construct the class instance from an Excel file.

        Parameters
        ----------
        file_path : str
            Path to the Excel file.
        y_column : str, default "y_t"
            Name of the column containing the time series.
        time_column : str or None, default "time"
            Name of the time column. If None, no time index is used.
        random_state : int, optional
            Seed for NumPy's RNG.

        Returns
        -------
        BayesianARModelSelector
        """
        data = pd.read_excel(file_path)
        y = data[y_column].to_numpy()

        if time_column is not None and time_column in data.columns:
            time = data[time_column].to_numpy() if time_column is not None and time_column in data.columns else time = None
   
        return cls(y=y, time=time, random_state=random_state)


    def plot_series_and_histogram(self) -> None:
        """
        Plot the trace and histogram of the effective sample y_t.
        """
        # Trace plot
        if self.time_eff is not None:
            x_axis = self.time_eff
            x_label = "Time"
        else:
            x_axis = np.arange(len(self.y_eff))
            x_label = "Index"

        plt.figure(figsize=(8, 6))
        plt.plot(x_axis, self.y_eff)
        plt.title("Trace plot: y_t over time")
        plt.xlabel(x_label)
        plt.ylabel("y_t")
        plt.tight_layout()
        plt.show()

        # Histogram with KDE
        plt.figure(figsize=(8, 6))
        sns.histplot(self.y_eff, kde=True, edgecolor="black")
        if plt.gca().lines:
            plt.gca().lines[0].set_linewidth(2.0)
        plt.title("Histogram of y_t")
        plt.xlabel("y_t")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _ols_fit(y: np.ndarray,X: np.ndarray,df: int,forecast_lags: Tuple[float, ...]) -> Tuple[float, np.ndarray, np.ndarray, float, float, float]:
        
        """
        Internal helper to run OLS and compute one-step-ahead prediction.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (T,).
        X : np.ndarray
            Design matrix (T x k).
        df : int
            Degrees of freedom (T - k).
        forecast_lags : tuple of floats
            Last observed lag values used for one-step-ahead forecast.
            For AR(1): (y_T,)
            For AR(2): (y_T, y_{T-1})

        Returns
        -------
        y_pred : float
            One-step-ahead prediction.
        beta_hat : np.ndarray
            OLS coefficients.
        vcov : np.ndarray
            Variance–covariance matrix of coefficients.
        sigma2 : float
            Residual variance estimate.
        lower : float
            Lower bound of 95% prediction interval.
        upper : float
            Upper bound of 95% prediction interval.
        """
        xtx = X.T @ X
        xtx_inv = np.linalg.inv(xtx)
        xty = X.T @ y

        beta_hat = (xtx_inv @ xty).squeeze()

        # Residuals and variance
        residuals = y - X @ beta_hat
        sigma2 = float((residuals @ residuals) / df)
        s = np.sqrt(sigma2)

        vcov = sigma2 * xtx_inv

        # Forecast
        y_pred = float(beta_hat[0] + np.dot(beta_hat[1:], np.array(forecast_lags)))

        # 95% PI using residual std only (simple approximation)
        z = 1.96
        lower = y_pred - z * s
        upper = y_pred + z * s

        return y_pred, beta_hat, vcov, sigma2, lower, upper

    def fit_ar1(self) -> Dict[str, Any]:
        """
        Fit AR(1) via OLS and compute 1-step-ahead prediction.

        Model:
            y_t = β_0 + β_1 y_{t-1} + ε_t

        Returns
        -------
        results : dict
            Keys: 'y_pred', 'beta', 'vcov', 'sigma2', 'lower', 'upper'
        """
        y_full = self.y  
        y_last = float(y_full[-1])

        y_pred, beta, vcov, sigma2, lower, upper = self._ols_fit(y=self.y_eff,X=self.X_ar1,df=self.df_ar1,forecast_lags=(y_last,))

        self.beta_ar1 = beta
        self.vcov_ar1 = vcov
        self.sigma2_ar1 = sigma2

        return {"y_pred": y_pred,"beta": beta,"vcov": vcov,"sigma2": sigma2,"lower": lower,"upper": upper}

    def fit_ar2(self) -> Dict[str, Any]:
        """
        Fit AR(2) via OLS and compute 1-step-ahead prediction.

        Model:
            y_t = β_0 + β_1 y_{t-1} + β_2 y_{t-2} + ε_t

        Returns
        -------
        results : dict
            Keys: 'y_pred', 'beta', 'vcov', 'sigma2', 'lower', 'upper'
        """
        y_full = self.y
        y_last = float(y_full[-1])
        y_prev = float(y_full[-2])

        y_pred, beta, vcov, sigma2, lower, upper = self._ols_fit(y=self.y_eff,X=self.X_ar2,df=self.df_ar2,forecast_lags=(y_last, y_prev))

        self.beta_ar2 = beta
        self.vcov_ar2 = vcov
        self.sigma2_ar2 = sigma2

        return {"y_pred": y_pred,"beta": beta,"vcov": vcov,"sigma2": sigma2,"lower": lower,"upper": upper}

  

    def _bayesian_prediction(self,X: np.ndarray,beta_hat: np.ndarray,vcov: np.ndarray,df: int,forecast_lags: Tuple[float, ...],n_draws: int) -> Tuple[float, np.ndarray, float, float]:
        """
        Internal helper for Bayesian one-step-ahead prediction.

        Uses a Normal–Gamma style approximation:
        - Draw β from an approximate multivariate t using χ² + Normal.
        - Given β, draw precision from Gamma using squared residuals.
        - Simulate predictive draws for y_{T+1}.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (T x k).
        beta_hat : np.ndarray
            Point estimate of coefficients.
        vcov : np.ndarray
            Covariance matrix of coefficients.
        df : int
            Degrees of freedom used for β (χ²).
        forecast_lags : tuple of floats
            Lag values used for prediction.
        n_draws : int
            Number of Monte Carlo draws.

        Returns
        -------
        mean_pred : float
            Posterior mean of y_{T+1}.
        draws : np.ndarray
            Predictive draws.
        lower : float
            2.5% predictive quantile.
        upper : float
            97.5% predictive quantile.
        """
        y = self.y_eff
        T = len(y)
        k = X.shape[1]
        draws = []

        for _ in range(n_draws):
            # Draw β
            v_i = np.random.chisquare(df)
            w_i = np.random.multivariate_normal(np.zeros(k), vcov)
            beta_i = beta_hat + (1.0 / np.sqrt(v_i / df)) * w_i

            # Residuals for this β
            residuals = y - X @ beta_i
            scale = 0.5 * np.sum(residuals**2)

            # Precision ~ Gamma(shape=T/2, scale=1/scale)
            hi = np.random.gamma(T / 2.0, 1.0 / scale)

            # Predictive error
            error = np.random.normal(0.0, 1.0 / np.sqrt(hi))

            # y_{T+1}
            y_pred = beta_i[0] + np.dot(beta_i[1:], np.array(forecast_lags)) + error
            draws.append(y_pred)

        draws = np.asarray(draws)
        mean_pred = float(np.mean(draws))
        lower = float(np.quantile(draws, 0.025))
        upper = float(np.quantile(draws, 0.975))

        return mean_pred, draws, lower, upper

    def bayesian_prediction_ar1(self, n_draws: int = 10_000) -> Dict[str, Any]:
        """
        Bayesian one-step-ahead prediction for AR(1).

        Returns
        -------
        results : dict
            Keys: 'mean', 'draws', 'lower', 'upper'
        """
        if self.beta_ar1 is None or self.vcov_ar1 is None:
            self.fit_ar1()

        y_last = float(self.y[-1])

        mean_pred, draws, lower, upper = self._bayesian_prediction(X=self.X_ar1,beta_hat=self.beta_ar1,vcov=self.vcov_ar1,df=self.df_ar1,forecast_lags=(y_last,),n_draws=n_draws)

        return {"mean": mean_pred,"draws": draws,"lower": lower,"upper": upper}

    def bayesian_prediction_ar2(self, n_draws: int = 10_000) -> Dict[str, Any]:
        """
        Bayesian one-step-ahead prediction for AR(2).

        Returns
        -------
        results : dict
            Keys: 'mean', 'draws', 'lower', 'upper'
        """
        if self.beta_ar2 is None or self.vcov_ar2 is None:
            self.fit_ar2()

        y_last = float(self.y[-1])
        y_prev = float(self.y[-2])

        mean_pred, draws, lower, upper = self._bayesian_prediction(X=self.X_ar2,beta_hat=self.beta_ar2,vcov=self.vcov_ar2,df=self.df_ar2,forecast_lags=(y_last, y_prev),n_draws=n_draws)

        return {"mean": mean_pred, "draws": draws,"lower": lower,"upper": upper}

   

    def build_ar2_prior(self, m: int) -> Dict[str, Any]:
        """
        Construct a prior for AR(2) coefficients from the first m
        effective observations.

        The prior is Normal with:
        - Mean = OLS estimate on the first m observations.
        - Covariance = s^2 * (X'X)^{-1} from the first m observations.
        Degrees of freedom for the prior t-marginal are df_prior = m - 3.

        Parameters
        ----------
        m : int
            Number of initial effective observations used (m >= 3).

        Returns
        -------
        results : dict
            Keys: 'beta_prior', 'vcov_prior', 'df_prior'
        """
        if m < 3:
            raise ValueError("m must be at least 3 for AR(2) prior (3 parameters).")

        if m > len(self.y_eff):
            raise ValueError("m cannot exceed the number of effective observations.")

        y_sub = self.y_eff[:m]
        X_sub = self.X_ar2[:m, :]

        xtx = X_sub.T @ X_sub
        xtx_inv = np.linalg.inv(xtx)
        xty = X_sub.T @ y_sub

        beta_prior = (xtx_inv @ xty).squeeze()
        residuals = y_sub - X_sub @ beta_prior
        df_prior = m - 3
        s2_prior = float((residuals @ residuals) / df_prior)
        vcov_prior = s2_prior * xtx_inv

        self.beta_prior_ar2 = beta_prior
        self.vcov_prior_ar2 = vcov_prior
        self.df_prior_ar2 = df_prior

        return {"beta_prior": beta_prior,"vcov_prior": vcov_prior,"df_prior": df_prior}

 

    def sddr_model_averaging(self,m: int,n_draws: int = 10_000,verbose: bool = True) -> Dict[str, Any]:
        """
        Perform SDDR-based model comparison and Bayesian model averaging.

        Steps
        -----
        1. Build AR(2) prior from first m effective observations.
        2. Compute marginal prior and posterior density of β_2 at 0.
        3. Savage–Dickey Bayes factor = posterior / prior (favoring AR(1)).
        4. Convert Bayes factor into posterior model probabilities.
        5. Draw predictive samples from AR(1) and AR(2) in proportion to
           their posterior probabilities.
        6. Return combined predictive summary.

        Parameters
        ----------
        m : int
            Number of effective observations used for the AR(2) prior.
        n_draws : int, default 10_000
            Total number of predictive draws for model averaging.
        verbose : bool, default True
            If True, prints Bayes factor and probabilities.

        Returns
        -------
        results : dict
            Keys:
            - 'bayes_factor'
            - 'prob_ar1'
            - 'prob_ar2'
            - 'combined_mean'
            - 'combined_lower'
            - 'combined_upper'
            - 'ar1_draws'
            - 'ar2_draws'
            - 'combined_draws'
        """
        # Ensure OLS fits exist
        if self.beta_ar1 is None or self.vcov_ar1 is None:
            self.fit_ar1()
        if self.beta_ar2 is None or self.vcov_ar2 is None:
            self.fit_ar2()

        # Build prior for AR(2)
        self.build_ar2_prior(m=m)
        if self.beta_prior_ar2 is None or self.vcov_prior_ar2 is None:
            raise RuntimeError("AR(2) prior not properly constructed.")

        # Posterior density of β_2 at 0 (Student-t marginal)
        posterior_density = t.pdf(0.0,df=self.df_ar2,loc=self.beta_ar2[2],scale=np.sqrt(self.vcov_ar2[2, 2]))

        # Prior density of β_2 at 0
        prior_density = t.pdf(0.0,df=self.df_prior_ar2,loc=self.beta_prior_ar2[2],scale=np.sqrt(self.vcov_prior_ar2[2, 2]))

        # Bayes factor in favor of AR(1) (β_2 = 0)
        bayes_factor = posterior_density / prior_density
        prob_ar1 = float(np.round(bayes_factor / (1.0 + bayes_factor), 4))
        prob_ar2 = float(np.round(1.0 - prob_ar1, 4))

        # Allocate draws
        n_draws_ar1 = int(prob_ar1 * n_draws)
        n_draws_ar2 = n_draws - n_draws_ar1

        # Predictive draws from each model
        ar1_pred = self.bayesian_prediction_ar1(n_draws=n_draws_ar1)
        ar2_pred = self.bayesian_prediction_ar2(n_draws=n_draws_ar2)

        ar1_draws = ar1_pred["draws"]
        ar2_draws = ar2_pred["draws"]

        combined_draws = np.concatenate([ar1_draws, ar2_draws])
        combined_mean = float(np.mean(combined_draws))
        combined_lower = float(np.quantile(combined_draws, 0.025))
        combined_upper = float(np.quantile(combined_draws, 0.975))

        if verbose:
            print(f"Posterior density at β₂ = 0 (AR(2)): {posterior_density:.5f}")
            print(f"Prior density at β₂ = 0 (AR(2)):     {prior_density:.5f}")
            print(f"Bayes factor in favor of AR(1):      {bayes_factor:.4f}")
            print(f"P(AR(1) | y): {prob_ar1:.4f}")
            print(f"P(AR(2) | y): {prob_ar2:.4f}")
            print()
            print(f"Combined predictive mean for y_(T+1): {combined_mean:.4f}")
            print(f"95% posterior predictive interval {combined_lower:.4f}, {combined_upper:.4f}")

        return {"bayes_factor": bayes_factor,"prob_ar1": prob_ar1,"prob_ar2": prob_ar2,"combined_mean": combined_mean,"combined_lower": combined_lower,"combined_upper": combined_upper,"ar1_draws": ar1_draws,"ar2_draws": ar2_draws,"combined_draws": combined_draws}


