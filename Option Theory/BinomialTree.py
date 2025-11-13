from typing import Literal
import numpy as np

class BinomialTreePricer:
    """Cox–Ross–Rubinstein tree. Prices European or American vanilla options."""

    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float, steps: int):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = steps

        self.dt = T / steps
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.df = np.exp(-r * self.dt)
        self.q = (np.exp(r * self.dt) - self.d) / (self.u - self.d)

        if not (0.0 < self.q < 1.0):
            raise ValueError("Risk-neutral probability q must lie in (0,1); check inputs.")

    def _terminal_payoff(self, option: Literal["call", "put"]) -> np.ndarray:
        j = np.arange(self.N + 1)
        ST = self.S0 * (self.u ** (self.N - j)) * (self.d ** j)
        if option == "call":
            return np.maximum(ST - self.K, 0.0)
        else:
            return np.maximum(self.K - ST, 0.0)

    def price(self, option: Literal["call", "put"], american: bool = False) -> float:
        V = self._terminal_payoff(option)
        for i in range(self.N - 1, -1, -1):
            V = self.df * (self.q * V[:-1] + (1.0 - self.q) * V[1:])
            if american:
                j = np.arange(i + 1)
                Sij = self.S0 * (self.u ** (i - j)) * (self.d ** j)
                if option == "call":
                    intrinsic = Sij - self.K
                else:
                    intrinsic = self.K - Sij
                V = np.maximum(V, intrinsic)
        return float(V[0])
