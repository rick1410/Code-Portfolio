from typing import Literal
import numpy as np

class AsianOptions:
    """Utilities for exotic payoffs using simulated paths."""

    @staticmethod
    def asian_call(S_paths: np.ndarray, K: float, r: float, T: float,
                   last_window_steps: int) -> float:
        if last_window_steps <= 0:
            raise ValueError("last_window_steps must be positive.")
        if last_window_steps > S_paths.shape[1]:
            raise ValueError("last_window_steps exceeds available path length.")

        window = S_paths[:, -last_window_steps:]
        avg_price = window.mean(axis=1)
        payoff = np.maximum(avg_price - K, 0.0)
        return float(np.exp(-r * T) * payoff.mean())

    @staticmethod
    def asian_put(S_paths: np.ndarray, K: float, r: float, T: float,
                  last_window_steps: int) -> float:
        if last_window_steps <= 0:
            raise ValueError("last_window_steps must be positive.")
        if last_window_steps > S_paths.shape[1]:
            raise ValueError("last_window_steps exceeds available path length.")

        window = S_paths[:, -last_window_steps:]
        avg_price = window.mean(axis=1)
        payoff = np.maximum(K - avg_price, 0.0)
        return float(np.exp(-r * T) * payoff.mean())
