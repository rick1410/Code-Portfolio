from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class GBMParams:
    mu: float
    sigma: float
    dt: float
    seed: int | None = 42

class GBMSimulator:
    """Simulate GBM paths with exact log-normal scheme."""
    def __init__(self, params: GBMParams):
        self.params = params
        self.rng = np.random.default_rng(self.params.seed)

    def paths(self, S0: float, steps: int, n_sims: int) -> np.ndarray:
        dt = self.params.dt
        mu = self.params.mu
        sigma = self.params.sigma

        Z = self.rng.standard_normal((n_sims, steps))
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.sqrt(dt) * Z
        log_increments = drift + shock
        log_levels = np.cumsum(log_increments, axis=1)
        S = S0 * np.exp(np.c_[np.zeros(n_sims), log_levels])
        return S

    @staticmethod
    def mc_vanilla_price(S_paths: np.ndarray, K: float, r: float, T: float,
                         option: Literal["call", "put"] = "call") -> float:
        ST = S_paths[:, -1]
        if option == "call":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)
        return float(np.exp(-r * T) * payoff.mean())

@dataclass(frozen=True)
class MCSpec:
    steps: int
    n_sims: int

class EulerGBMSimulator:
    """Euler–Maruyama simulator for GBM."""
    def __init__(self, params: GBMParams):
        self.params = params
        self.rng = np.random.default_rng(self.params.seed)

    def paths(self, S0: float, steps: int, n_sims: int) -> np.ndarray:
        dt = self.params.dt
        mu = self.params.mu
        sigma = self.params.sigma

        S = np.empty((n_sims, steps + 1), dtype=float)
        S[:, 0] = S0
        sqrt_dt = np.sqrt(dt)
        for n in range(steps):
            Z = self.rng.standard_normal(n_sims)
            S[:, n + 1] = S[:, n] + S[:, n] * (mu * dt + sigma * sqrt_dt * Z)
        return S
