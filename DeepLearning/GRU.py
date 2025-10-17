from typing import Any, Tuple
import torch
import torch.nn as nn
from .SelfAttention import Attention

class MultiStepGRU(nn.Module):
    """
    Multi-step GRU forecaster.

    A stacked GRU encoder that maps an input sequence (batch, T, n_feat)
    to n_steps parallel forecasts. The last hidden state is projected to
    the n_steps outputs, then unsqueezed to shape (batch, n_steps, 1).

    Parameters
    ----------
    n_feat : int
        Number of input features per time step.
    hidden : int, default 8
        Hidden size of the GRU.
    n_layers : int, default 2
        Number of GRU layers.
    n_steps : int, default 1
        Number of forecast steps.

    Attributes
    ----------
    n_steps : int
        Number of forecast steps.
    hidden_size : int
        Hidden size used by the GRU.
    n_layers : int
        Number of GRU layers.
    gru : nn.GRU
        Encoder GRU.
    fc : nn.Linear
        Output head projecting last hidden state to n_steps.
    """

    n_steps: int
    hidden_size: int
    n_layers: int
    gru: nn.GRU
    fc: nn.Linear

    def __init__(self, n_feat: int, hidden: int = 8, n_layers: int = 2, n_steps: int = 1) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.hidden_size = hidden
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size=n_feat, hidden_size=hidden, num_layers=n_layers, batch_first=True, dropout=0.0 if n_layers == 1 else 0.2)
        self.fc = nn.Linear(hidden, n_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, n_feat).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, n_steps, 1).
        """
        batch = x.size(0)
        h0 = torch.zeros(self.n_layers, batch, self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        last = out[:, -1, :]
        return self.fc(last).unsqueeze(-1)

def build_gru(evaluator: Any) -> MultiStepGRU:
    """
    Convenience constructor for `MultiStepGRU`.

    Parameters
    ----------
    evaluator : Any
        Object providing `X_seq` (to infer n_feat) and `horizons` (to infer n_steps).

    Returns
    -------
    MultiStepGRU
        Instantiated model with preset hyperparameters.
    """
    n_feat = int(evaluator.X_seq.shape[2])
    n_steps = int(len(evaluator.horizons))
    return MultiStepGRU(n_feat, hidden=8, n_layers=2, n_steps=n_steps)

class MultiStepGRUDecoder(nn.Module):
    """
    Attention-equipped GRU decoder for multi-step forecasting.

    Iteratively decodes `n_steps` outputs using additive attention over
    encoder outputs and a GRU cell conditioned on context.

    Parameters
    ----------
    hidden : int
        Hidden size (must match encoder hidden size).
    dropout : float
        Unused here, kept for API parity.
    n_steps : int
        Number of decoding steps.

    Attributes
    ----------
    n_steps : int
        Number of decoding steps.
    attn : Attention
        Additive attention module.
    gru : nn.GRU
        GRU taking concatenated [context, state] as input.
    head : nn.Sequential
        Projection head producing a scalar per step.
    """

    n_steps: int
    attn: Attention
    gru: nn.GRU
    head: nn.Sequential

    def __init__(self, hidden: int, dropout: float, n_steps: int) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.attn = Attention(hidden)
        self.gru = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, enc_outputs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode multi-step outputs.

        Parameters
        ----------
        enc_outputs : torch.Tensor
            Encoder outputs of shape (batch, seq_len, hidden).
        h_prev : torch.Tensor
            Previous hidden state of shape (1, batch, hidden).

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            Tuple of (predictions (batch, n_steps, 1), final_hidden (1, batch, hidden)).
        """
        preds = []
        h = h_prev
        for _ in range(self.n_steps):
            dec_state = h.squeeze(0)
            context, _ = self.attn(dec_state, enc_outputs)
            dec_input = torch.cat([context, dec_state], dim=-1).unsqueeze(1)
            out, h = self.gru(dec_input, h)
            step_pred = self.head(out.squeeze(1))
            preds.append(step_pred)
        stacked = torch.cat(preds, dim=1)
        return stacked.unsqueeze(-1), h

class MultiStepGRUAttention(nn.Module):
    """
    Encoder–attention–decoder GRU for multi-step forecasting.

    Parameters
    ----------
    n_feat : int
        Number of input features per time step.
    hidden : int, default 32
        Hidden size of the encoder/decoder.
    n_layers : int, default 3
        Number of GRU layers in the encoder.
    dropout : float, default 0.2
        Dropout probability in the encoder GRU.
    n_steps : int, default 1
        Number of forecast steps.

    Attributes
    ----------
    n_steps : int
        Number of forecast steps.
    encoder : nn.GRU
        GRU encoder.
    decoder : MultiStepGRUDecoder
        Attention-equipped decoder.
    """

    n_steps: int
    encoder: nn.GRU
    decoder: MultiStepGRUDecoder

    def __init__(self, n_feat: int, hidden: int = 32, n_layers: int = 3, dropout: float = 0.2, n_steps: int = 1) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.encoder = nn.GRU(input_size=n_feat, hidden_size=hidden, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.decoder = MultiStepGRUDecoder(hidden, dropout, n_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, n_feat).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, n_steps, 1).
        """
        enc_outputs, h_n = self.encoder(x)
        h0 = h_n[-1:].contiguous()
        preds, _ = self.decoder(enc_outputs, h0)
        return preds

def build_gru_attention(evaluator: Any) -> MultiStepGRUAttention:
    """
    Convenience constructor for `MultiStepGRUAttention`.

    Parameters
    ----------
    evaluator : Any
        Object providing `X_seq` (to infer n_feat) and `horizons` (to infer n_steps).

    Returns
    -------
    MultiStepGRUAttention
        Instantiated attention GRU model with preset hyperparameters.
    """
    n_feat = int(evaluator.X_seq.shape[2])
    n_steps = int(len(evaluator.horizons))
    return MultiStepGRUAttention(n_feat, hidden=32, n_layers=2, dropout=0.2, n_steps=n_steps)
