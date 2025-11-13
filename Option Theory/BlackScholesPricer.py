from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.stats import norm
from typing import Literal, Optional
import numpy as np

@dataclass(frozen=True)
class BSParams:
    r: float
    sigma: float
    T: float
    S0: float
    K: float

class BlackScholesPricer:
    def __init__(self, params: BSParams):
        self.params = params

    def _d12(self) -> Tuple[float, float]:
        p = self.params
        d1 = (np.log(p.S0 / p.K) + (p.r + 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
        d2 = d1 - p.sigma * np.sqrt(p.T)
        return d1, d2

    def call(self) -> float:
        p = self.params
        d1, d2 = self._d12()
        return float(p.S0 * norm.cdf(d1) - p.K * np.exp(-p.r * p.T) * norm.cdf(d2))

    def put(self) -> float:
        p = self.params
        d1, d2 = self._d12()
        return float(p.K * np.exp(-p.r * p.T) * norm.cdf(-d2) - p.S0 * norm.cdf(-d1))



class MonteCarloTerminalPricer:
    """Direct terminal-draw Monte Carlo for European options under GBM."""
    @staticmethod
    def european_price(S0: float, K: float, r: float, sigma: float, T: float,
                       n_sims: int, option: Literal["call", "put"] = "call",
                       seed: Optional[int] = None) -> float:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(n_sims)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        if option == "call":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)
        return float(np.exp(-r * T) * payoff.mean())

