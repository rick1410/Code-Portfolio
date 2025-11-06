from typing import Any, Sequence
import torch
import torch.nn as nn

class SimpleFFN(nn.Module):
    """
    Simple feedforward neural network for multi-horizon regression.

    The network flattens an input tensor of shape (batch, T, F) and maps it to
    a vector of predictions with length equal to the number of horizons, then
    unsqueezes to shape (batch, horizons, 1) to align with downstream evaluators.

    Parameters
    ----------
    evaluator : Any
        Object providing `X_seq` (used for input dimensionality) and `horizons` (list of forecast horizons).
        Expected: `evaluator.X_seq.shape` is (N, T, F) and `evaluator.horizons` is a Sequence.
    h1 : int, default 64
        Hidden size of the first fully connected layer.
    h2 : int, default 32
        Hidden size of the second fully connected layer.
    drop : float, default 0.1
        Dropout probability applied after each hidden layer.

    Attributes
    ----------
    horizons : Sequence[int] | Sequence[float]
        Forecast horizons taken from `evaluator.horizons`.
    net : nn.Sequential
        The feedforward network: Flatten -> Linear(h1) -> ReLU -> Dropout -> Linear(h2) -> ReLU -> Dropout -> Linear(out).
    """

    horizons: Sequence[int] | Sequence[float]
    net: nn.Sequential

    def __init__(self, evaluator: Any, h1: int = 64, h2: int = 32, drop: float = 0.1) -> None:
        super().__init__()
        n_in = int(evaluator.X_seq.shape[1] * evaluator.X_seq.shape[2])
        self.horizons = evaluator.horizons
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_in, h1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h2, len(self.horizons))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, T, F).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, horizons, 1).
        """
        return self.net(x).unsqueeze(-1)

def build_feedforward(evaluator: Any) -> SimpleFFN:
    """
    Convenience builder for SimpleFFN with preset hyperparameters.

    Parameters
    ----------
    evaluator : Any
        Object providing `X_seq` and `horizons`.

    Returns
    -------
    SimpleFFN
        Instantiated feedforward model.
    """
    return SimpleFFN(evaluator, h1=48, h2=24, drop=0.2)
