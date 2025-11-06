from scipy.special import gamma, digamma, polygamma
import numpy as np

class PDFModels:
  
    @staticmethod
    def ast_pdf(x, mu_t, h_t, alpha, nu1, nu2):
        
        # Precompute gamma-related constants
        K_nu1 = gamma((nu1 + 1) / 2.0) / (np.sqrt(np.pi * nu1) * gamma(nu1 / 2.0))
        K_nu2 = gamma((nu2 + 1) / 2.0) / (np.sqrt(np.pi * nu2) * gamma(nu2 / 2.0))

        B = alpha * K_nu1 + (1 - alpha) * K_nu2
        alpha_star = (alpha * K_nu1) / B

        # Compute m and s
        num_left  = alpha_star**2 * (nu1 / (nu1 - 1.0))
        num_right = (1.0 - alpha_star)**2 * (nu2 / (nu2 - 1.0))
        m = 4.0 * B * ( - num_left + num_right )

        var_left  = alpha_star**2 * (nu1 / (nu1 - 2.0))
        var_right = (1.0 - alpha_star)**2 * (nu2 / (nu2 - 2.0))
        s = np.sqrt(4.0 * (alpha * var_left + (1.0 - alpha) * var_right) - m**2)

        # Common repeated terms
        sqrt_h_t = np.sqrt(h_t)
        z = (x - mu_t) / sqrt_h_t   # standard scaling
        tmp = (m + s*z) / (2.0 * alpha_star)  # used in left_tail and right_tail
        tmp_sq = tmp * tmp

        # Left & right tail
        # Note: For z <= 0 (scaled_z <= 0), we use nu1; else use nu2
        left_tail_exponent = - (nu1 + 1.0) / 2.0
        right_tail_exponent = - (nu2 + 1.0) / 2.0

        left_term = 1.0 + (1.0 / nu1) * tmp_sq
        right_term = 1.0 + (1.0 / nu2) * tmp_sq

        left_tail = np.power(left_term,  left_tail_exponent)
        right_tail = np.power(right_term, right_tail_exponent)

        scaled_z = m + s*z
        mask_right = scaled_z > 0.0

        pdf = np.where(mask_right,right_tail,left_tail)
        pdf *= (s * B) / sqrt_h_t

        return pdf

    @staticmethod
    def egb2_pdf(x, mu_t, h_t, p, q):
        
        sqrt_h_t = np.sqrt(h_t)
        delta = digamma(p) - digamma(q)
        omega = polygamma(0, p) + polygamma(0, q)

        sqrt_omega = np.sqrt(omega)
        term = sqrt_omega * ((x - mu_t) / sqrt_h_t) + delta
        gamma_factor = (gamma(p) * gamma(q)) / gamma(p + q)

        numerator = sqrt_omega * np.exp(p * term)
        denominator = sqrt_h_t * gamma_factor * np.power(1.0 + np.exp(term), p + q)
        pdf = numerator / denominator

        return pdf

    @staticmethod
    def student_t_pdf(x, mu_t, h_t, v):
     
        gamma_ratio = gamma((v + 1.0) / 2.0) / gamma(v / 2.0)
        denom = np.sqrt(v * np.pi) * np.sqrt(h_t)
        prefactor = gamma_ratio / denom
        z = (x - mu_t) / np.sqrt(h_t)
        term = 1.0 + (z**2) / v

        exponent = - (v + 1.0) / 2.0
        pdf = prefactor * np.power(term, exponent)
        return pdf
