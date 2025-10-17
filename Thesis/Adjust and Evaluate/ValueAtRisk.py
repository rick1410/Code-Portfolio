from typing import Any, Dict, Tuple
import numpy as np
from scipy.stats import chi2
import scipy.stats as stats


class ValueAtRisk:
    """
    Value-at-Risk (VaR) utilities for several innovation distributions.

    Parameters
    ----------
    pdf_model : Any
        Object providing PDF evaluators used for quantile inversion via
        numerical integration. Must implement:
          - `egb2_pdf(x, mu, sigma2, p, q)`
          - `ast_pdf(x, mu, sigma2, delta, v1, v2)`

    Attributes
    ----------
    pdf_model : Any
        Stored PDF model used in quantile calculations.
    """

    pdf_model: Any

    def __init__(self, pdf_model: Any) -> None:
        self.pdf_model = pdf_model

    def calculate_quantile(self, distribution: str, alpha: float, params: Dict[str, float], sigma_forecast: float) -> float:
        """
        Numerically approximate the alpha-quantile of standardized returns.

        Uses a uniform grid on [-20, 20], integrates the PDF to a CDF via
        cumulative sums, then inverts the CDF at probability `alpha`.

        Parameters
        ----------
        distribution : str
            One of {"EGB2","AST"}.
        alpha : float
            Lower-tail probability (e.g., 0.01 for 1%).
        params : dict
            Distribution parameters; expected keys:
              - EGB2: {"p","q"}
              - AST : {"delta","v1","v2"}
        sigma_forecast : float
            Conditional variance forecast (used by PDFs).

        Returns
        -------
        float
            Quantile of the standardized return (mean 0, variance `sigma_forecast`).
        """
        dLBofGrid = -20.0; dUBofGrid = 20.0; dDistanceBetweenGridPoints = 0.001
        vGridCenters = np.arange(dLBofGrid + 0.5 * dDistanceBetweenGridPoints, dUBofGrid, dDistanceBetweenGridPoints)
        mu_t = 0.0

        if distribution == "EGB2":
            p = params["p"]; q = params["q"]
            vPdf = self.pdf_model.egb2_pdf(vGridCenters, mu_t, sigma_forecast, p, q)
        elif distribution == "AST":
            alpha_param = params["delta"]; nu1 = params["v1"]; nu2 = params["v2"]
            vPdf = self.pdf_model.ast_pdf(vGridCenters, mu_t, sigma_forecast, alpha_param, nu1, nu2)
        else:
            raise ValueError(f"Unsupported distribution '{distribution}'. Expected 'EGB2' or 'AST'.")

        vCDF = np.cumsum(vPdf) * dDistanceBetweenGridPoints
        if alpha > float(vCDF[-1]):
            quantile_index = len(vGridCenters) - 1
        else:
            quantile_index = int(np.searchsorted(vCDF, alpha))
        return float(vGridCenters[quantile_index])

    def calculate_var(self, model: Any, sigma_forecast: float, confidence_level: float) -> float:
        """
        Compute one-step VaR given a fitted volatility model and variance forecast.

        Parameters
        ----------
        model : Any
            Fitted model with attributes:
              - `distribution` in {"Normal","Student t","AST","EGB2"}
              - `optimal_params` (dict) including "nu1" for Student t.
        sigma_forecast : float
            Forecast variance for the next step.
        confidence_level : float
            VaR confidence (e.g., 0.99 for 1% left tail).

        Returns
        -------
        float
            VaR level (left tail), same scale as returns.
        """
        alpha = 1.0 - confidence_level
        mu_t = 0.0

        if model.distribution == "Normal":
            z_alpha = stats.norm.ppf(alpha)
            VaR = mu_t + z_alpha * np.sqrt(sigma_forecast)
        elif model.distribution == "Student t":
            nu = float(model.optimal_params["nu1"])
            t_alpha = stats.t.ppf(alpha, df=nu) * np.sqrt((nu - 2.0) / nu)
            VaR = mu_t + t_alpha * np.sqrt(sigma_forecast)
        elif model.distribution == "AST":
            AST_alpha = self.calculate_quantile("AST", alpha, model.optimal_params, sigma_forecast)
            VaR = mu_t + AST_alpha * np.sqrt(sigma_forecast)
        elif model.distribution == "EGB2":
            EGB2_alpha = self.calculate_quantile("EGB2", alpha, model.optimal_params, sigma_forecast)
            VaR = mu_t + EGB2_alpha * np.sqrt(sigma_forecast)
        else:
            raise ValueError(f"Unsupported model distribution '{model.distribution}'.")
        return float(VaR)

    @staticmethod
    def calculate_var_ML(y_hat_next: float, confidence_level: float, log_returns_train: np.ndarray, realised_var_train: np.ndarray) -> float:
        """
        ML-based VaR using empirical residual quantile.

        Parameters
        ----------
        y_hat_next : float
            Forecast of next-period variance (or volatility proxy squared).
        confidence_level : float
            VaR confidence (e.g., 0.99 -> alpha = 0.01).
        log_returns_train : np.ndarray
            In-sample returns for residual construction.
        realised_var_train : np.ndarray
            In-sample realized variance aligned with returns.

        Returns
        -------
        float
            VaR estimate using empirical epsilon quantile scaled by sqrt(y_hat_next).
        """
        alpha = 1.0 - confidence_level
        eps = np.asarray(log_returns_train) / np.sqrt(np.asarray(realised_var_train))
        q_alpha = float(np.quantile(eps, alpha))
        VaR = float(np.sqrt(y_hat_next) * q_alpha)
        return VaR

    @staticmethod
    def backtest_var(v_indicator_violations: np.ndarray, prob_var_violation_under_h0: float) -> Tuple[float, float, float, float, float, float]:
        """
        VaR backtests: Unconditional Coverage (UC), Conditional Coverage (CC), and Independence (IND).

        Parameters
        ----------
        v_indicator_violations : np.ndarray
            1-D array of 0/1 indicators: 1 if return < VaR (violation), else 0.
        prob_var_violation_under_h0 : float
            Nominal violation probability under H0 (e.g., alpha = 0.01).

        Returns
        -------
        (float, float, float, float, float, float)
            (LR_UC, p_UC, LR_CC, p_CC, LR_IND, p_IND).
        """
        v = np.asarray(v_indicator_violations).astype(int)
        n_total = int(len(v)); n = n_total - 1

        i_n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
        i_n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
        i_n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
        i_n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

        i_n0 = i_n00 + i_n10
        i_n1 = i_n01 + i_n11

        d_logl_h0 = i_n0 * np.log(max(1e-300, 1.0 - prob_var_violation_under_h0)) + i_n1 * np.log(max(1e-300, prob_var_violation_under_h0))
        d_logl_h1_uc = 0.0
        if i_n0 > 0:
            d_logl_h1_uc += i_n0 * np.log(max(1e-300, i_n0 / n))
        if i_n1 > 0:
            d_logl_h1_uc += i_n1 * np.log(max(1e-300, i_n1 / n))

        d_logl_h1_cc = 0.0
        if i_n00 > 0:
            d_logl_h1_cc += i_n00 * np.log(max(1e-300, i_n00 / (i_n00 + i_n01)))
        if i_n01 > 0:
            d_logl_h1_cc += i_n01 * np.log(max(1e-300, i_n01 / (i_n00 + i_n01)))
        if i_n10 > 0:
            d_logl_h1_cc += i_n10 * np.log(max(1e-300, i_n10 / (i_n10 + i_n11)))
        if i_n11 > 0:
            d_logl_h1_cc += i_n11 * np.log(max(1e-300, i_n11 / (i_n10 + i_n11)))

        dLR_UC_VaR = 2.0 * (d_logl_h1_uc - d_logl_h0)
        dLR_CC_VaR = 2.0 * (d_logl_h1_cc - d_logl_h0)
        dLR_IND_VaR = dLR_CC_VaR - dLR_UC_VaR

        dPvalue_UC_VaR = 1.0 - chi2.cdf(dLR_UC_VaR, 1)
        dPvalue_CC_VaR = 1.0 - chi2.cdf(dLR_CC_VaR, 2)
        dPvalue_IND_VaR = 1.0 - chi2.cdf(dLR_IND_VaR, 1)

        return float(dLR_UC_VaR), float(dPvalue_UC_VaR), float(dLR_CC_VaR), float(dPvalue_CC_VaR), float(dLR_IND_VaR), float(dPvalue_IND_VaR)
