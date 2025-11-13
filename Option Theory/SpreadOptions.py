   
@dataclass
class SpreadOptionParams:
    S1_0: float
    S2_0: float
    sigma1: float
    sigma2: float
    rho: float
    T: float
    K: float = 0.0
    r: float = 0.0
    num_simulations: int = 10_000
    random_state: Optional[int] = None


class SpreadOptionMC:
    """
    Monte Carlo toolkit for pricing a (S1 - S2 - K)^+ spread option and related diagnostics.

    Notes
    -----
    - The Margrabe (exchange option) formula implemented here assumes K=0.
    - Discounting uses a continuously-compounded risk-free rate r.
    - For correlation sweeps, we reuse the same pair of standard normal draws
      by default so that price differences are driven only by changes in rho.
    """

    def __init__(self, params: SpreadOptionParams):
        self.params = params
        self._rng = np.random.default_rng(params.random_state)

        self._x1: Optional[np.ndarray] = None
        self._x2: Optional[np.ndarray] = None
        self._last_S1_T: Optional[np.ndarray] = None
        self._last_S2_T: Optional[np.ndarray] = None
        self._last_spread: Optional[np.ndarray] = None
        self._last_payoffs: Optional[np.ndarray] = None

    def _ensure_normals(self, force_new: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        n = self.params.num_simulations
        if force_new or self._x1 is None or self._x1.size != n:
            self._x1 = self._rng.standard_normal(n)
            self._x2 = self._rng.standard_normal(n)
        return self._x1, self._x2

    @staticmethod
    def _correlate(x1: np.ndarray, x2: np.ndarray, rho: float) -> Tuple[np.ndarray, np.ndarray]:
        eps1 = x1
        eps2 = rho * x1 + np.sqrt(max(0.0, 1.0 - rho**2)) * x2
        return eps1, eps2

    @property
    def implied_spread_vol(self) -> float:
        p = self.params
        return float(np.sqrt(p.sigma1**2 + p.sigma2**2 - 2.0 * p.rho * p.sigma1 * p.sigma2))

    def simulate_spread_price(self, plot: bool = True, bins: int = 50,filename: Optional[str] = "Distribution_of_Spread_at_Maturity.png",reuse_normals: bool = True) -> Dict[str, float]:
        """
        Price the spread call via bivariate Monte Carlo for current params.

        Returns a dict with the Monte Carlo price and σ̂ (implied spread vol).
        """
        p = self.params
        x1, x2 = self._ensure_normals(force_new=not reuse_normals)
        eps1, eps2 = self._correlate(x1, x2, p.rho)

        drift1 = -0.5 * p.sigma1**2 * p.T
        drift2 = -0.5 * p.sigma2**2 * p.T
        scale1 = p.sigma1 * np.sqrt(p.T)
        scale2 = p.sigma2 * np.sqrt(p.T)

        S1_T = p.S1_0 * np.exp(drift1 + scale1 * eps1)
        S2_T = p.S2_0 * np.exp(drift2 + scale2 * eps2)

        spread = S1_T - S2_T
        payoffs = np.maximum(spread - p.K, 0.0)
        disc = np.exp(-p.r * p.T)
        price = disc * float(np.mean(payoffs))

        # Cache latest arrays
        self._last_S1_T, self._last_S2_T = S1_T, S2_T
        self._last_spread, self._last_payoffs = spread, payoffs


        return {"mc_price": price,"sigma_hat": self.implied_spread_vol,"discount_factor": disc}

    def price_vs_correlation(self, rho_values: Optional[Sequence[float]] = None,plot: bool = True,filename: Optional[str] = None ,reuse_normals: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep correlation and compute MC prices at each ρ, reusing the same normal draws
        by default to isolate the impact of correlation.
        """
        p = self.params
        if rho_values is None:
            rho_values = np.linspace(-1.0, 1.0, 50)
        rho_values = np.asarray(rho_values, dtype=float)

        x1, x2 = self._ensure_normals(force_new=not reuse_normals)

        prices = np.empty_like(rho_values)
        drift1 = -0.5 * p.sigma1**2 * p.T
        drift2 = -0.5 * p.sigma2**2 * p.T
        scale1 = p.sigma1 * np.sqrt(p.T)
        scale2 = p.sigma2 * np.sqrt(p.T)
        disc = np.exp(-p.r * p.T)

        for i, rho in enumerate(rho_values):
            eps1, eps2 = self._correlate(x1, x2, rho)
            S1_T = p.S1_0 * np.exp(drift1 + scale1 * eps1)
            S2_T = p.S2_0 * np.exp(drift2 + scale2 * eps2)
            spread = S1_T - S2_T
            payoffs = np.maximum(spread - p.K, 0.0)
            prices[i] = disc * np.mean(payoffs)

        return rho_values, prices

    def margrabe_price(self) -> float:
        """
        Margrabe (1978) exchange option price: E[(S1_T - S2_T)^+] in BS world with K=0.
        If K>0, this formula does not apply.
        """
        p = self.params
        if p.K != 0:
            raise ValueError("Margrabe formula assumes K=0.")
        sigma_hat = self.implied_spread_vol
        if sigma_hat <= 0:
            return max(0.0, p.S1_0 - p.S2_0)  # degenerate vol

        d1 = (np.log(p.S1_0 / p.S2_0) + 0.5 * sigma_hat**2 * p.T) / (sigma_hat * np.sqrt(p.T))
        d2 = d1 - sigma_hat * np.sqrt(p.T)
       
        # Prices are spot; discounting cancels if both are forwards; we keep r for completeness
        price = p.S1_0 * norm.cdf(d1) - p.S2_0 * norm.cdf(d2)
        return float(np.exp(-p.r * p.T) * price)

    def gbm_spread_sim_price(self, reuse_normals: bool = True, plot: bool = True,bins: int = 50, filename: Optional[str] = None) -> float:
        """
        Treat the spread S = S1 - S2 as its own GBM with σ̂ and price the call via MC.
        Uses the object-level x1 normals by default for comparability.
        """
        p = self.params
        sigma_hat = self.implied_spread_vol
        if sigma_hat <= 0:
            return max(0.0, (p.S1_0 - p.S2_0) - p.K)

        x1, _ = self._ensure_normals(force_new=not reuse_normals)
        drift = -0.5 * sigma_hat**2 * p.T
        scale = sigma_hat * np.sqrt(p.T)

        S0 = p.S1_0 - p.S2_0
        S_T = S0 * np.exp(drift + scale * x1)
        payoffs = np.maximum(S_T - p.K, 0.0)
        price = float(np.exp(-p.r * p.T) * np.mean(payoffs))


        return price