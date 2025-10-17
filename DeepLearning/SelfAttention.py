from typing import Tuple
import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Bahdanau additive attention over encoder hidden states.

    Computes attention weights for each encoder timestep given the current
    decoder state, and returns the context vector as the weighted sum of
    encoder outputs.

    Parameters
    ----------
    hidden : int
        Hidden dimensionality for both encoder outputs and decoder state.

    Attributes
    ----------
    W : nn.Linear
        Linear projection applied to encoder outputs (no bias).
    v : nn.Linear
        Final linear scorer mapping tanh-projected vectors to a scalar (no bias).
    """

    W: nn.Linear
    v: nn.Linear

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute context vector and attention weights.

        Parameters
        ----------
        decoder_state : torch.Tensor
            Current decoder hidden state of shape (batch, hidden).
        encoder_outputs : torch.Tensor
            Encoder hidden states of shape (batch, seq_len, hidden).

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Tuple `(context, weights)` where:
              - `context` has shape (batch, hidden),
              - `weights` has shape (batch, seq_len) and sums to 1 over `seq_len`.
        """
        scores = self.v(torch.tanh(self.W(encoder_outputs) + decoder_state[:, None, :])).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = (weights[:, :, None] * encoder_outputs).sum(dim=1)
        return context, weights
