

class ImportanceSampling:
    """
    Wrapper class for importance sampling functions.

    All methods are static and preserve the original code exactly.
    """

    @staticmethod
    def ImportanceSamplingStudentT(data, y_bar, sample_var, n, n_draws):
        DoF_draw = [random.uniform(4.1, 50) for _ in range(n_draws)]
        y_given_DoF = []
        for i in range(n_draws):
            dummy_gamma = gamma(((DoF_draw[i] + 1) / 2)) / gamma((DoF_draw[i] / 2))
            scaler = 1 / (np.sqrt((DoF_draw[i] - 2) * np.pi))
            dummy_1 = (dummy_gamma * scaler) ** n
            var_scaler = 1 / ((sample_var) ** (n / 2))
            dummy_2 = dummy_1 * var_scaler
            power = -((DoF_draw[i] + 1) / 2)
            dummy_3 = (1 + ((data - y_bar) ** 2 / ((DoF_draw[i] - 2) * sample_var)))
            prod_part = np.prod(dummy_3 ** power)
            result = dummy_2 * prod_part
            y_given_DoF.append(result)

        marginal_likelihood_t = np.mean(y_given_DoF)
        return marginal_likelihood_t, DoF_draw, y_given_DoF

    @staticmethod
    def ImportanceSamplingGED(data, y_bar, sample_var, n, ndraws):
        beta_draw = [random.uniform(0.38, 1.88) for _ in range(n_draws)]
        y_given_beta = []
        for i in range(n_draws):
            dummy_1 = (beta_draw[i] * np.sqrt(gamma((3 / beta_draw[i]))))
            dummy_2 = (2 * np.sqrt(sample_var)) * ((gamma((1 / beta_draw[i]))) ** (3 / 2))
            dummy_first = (dummy_1 / dummy_2) ** n
            dummy_3 = (np.abs(data - y_bar))
            dummy_4 = (np.sqrt(sample_var)) * np.sqrt(
                (gamma((1 / beta_draw[i])) / gamma((3 / beta_draw[i])))
            )
            sum_term = np.sum(((dummy_3 / dummy_4) ** beta_draw[i]))
            exp = np.exp(-sum_term)
            result = dummy_first * exp
            y_given_beta.append(result)

        marginal_likelihood_GED = np.mean(y_given_beta)
        return marginal_likelihood_GED, beta_draw, y_given_beta

    @staticmethod
    def NewImportanceSamplingStudentT(data, y_bar, sample_var, n, n_draws):
        DoF_draws = [random.uniform(4.1, 1000) for _ in range(n_draws)]
        marginal_likelihoods = []
        for i in range(n_draws):
            help_1 = (gammaln(((DoF_draws[i] + 1) / 2))) - (gammaln((DoF_draws[i] / 2)))
            help_2 = (np.exp(help_1))
            help_3 = (1 / (np.sqrt((DoF_draws[i] - 2) * np.pi)))
            first_term = (help_2 * help_3) ** n
            second_term = first_term * (1 / ((sample_var) ** (n / 2)))
            help_power = -((DoF_draws[i] + 1) / 2)
            help_3 = (1 + ((data - y_bar) ** 2 / ((DoF_draws[i] - 2) * sample_var)))
            prod_term = np.prod(help_3 ** help_power)
            all_terms = second_term * prod_term
            marginal_likelihoods.append(all_terms)

        new_marginal_likelihood_t = np.mean(marginal_likelihoods)
        return new_marginal_likelihood_t, DoF_draws, marginal_likelihoods