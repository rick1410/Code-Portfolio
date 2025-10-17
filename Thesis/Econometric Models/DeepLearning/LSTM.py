from typing import Any, Tuple
import torch
import torch.nn as nn
from .SelfAttention import Attention

class MultiStepLSTM(nn.Module):
    """
    Multi-step LSTM forecaster.

    The network encodes an input sequence (batch, T, n_feat) with an LSTM,
    projects the last hidden state to `n_steps` parallel forecasts, then
    unsqueezes to shape (batch, n_steps, 1).

    Parameters
    ----------
    n_feat : int
        Number of input features per time step.
    hidden : int, default 8
        Hidden size of the LSTM.
    n_layers : int, default 2
        Number of LSTM layers.
    n_steps : int, default 1
        Number of forecast steps.

    Attributes
    ----------
    n_steps : int
        Number of forecast steps.
    hidden_size : int
        Hidden size used by the LSTM.
    n_layers : int
        Number of LSTM layers.
    lstm : nn.LSTM
        Encoder LSTM.
    fc : nn.Linear
        Output head projecting last hidden state to n_steps.
    """

    n_steps: int
    hidden_size: int
    n_layers: int
    lstm: nn.LSTM
    fc: nn.Linear

    def __init__(self, n_feat: int, hidden: int = 8, n_layers: int = 2, n_steps: int = 1) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.hidden_size = hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=n_layers, batch_first=True, dropout=0.2)
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
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        last = out[:, -1, :]
        return self.fc(last).unsqueeze(-1)

def build_lstm(evaluator: Any) -> MultiStepLSTM:
    """
    Convenience constructor for `MultiStepLSTM`.

    Parameters
    ----------
    evaluator : Any
        Object providing `X_seq` (to infer n_feat) and `horizons` (to infer n_steps).

    Returns
    -------
    MultiStepLSTM
        Instantiated model with preset hyperparameters.
    """
    n_feat = int(evaluator.X_seq.shape[2])
    n_steps = int(len(evaluator.horizons))
    return MultiStepLSTM(n_feat, hidden=8, n_layers=2, n_steps=n_steps)

class MultiStepLSTMDecoder(nn.Module):
    """
    Attention-equipped LSTM decoder for multi-step forecasting.

    Iteratively decodes `n_steps` outputs using additive attention over
    encoder outputs and an LSTM conditioned on the attention context.

    Parameters
    ----------
    hidden : int
        Hidden size (must match encoder hidden size).
    n_layers : int
        Number of layers in the decoder LSTM.
    dropout : float
        Dropout probability in the decoder LSTM.
    n_steps : int
        Number of decoding steps.

    Attributes
    ----------
    n_steps : int
        Number of decoding steps.
    attn : Attention
        Additive attention module.
    lstm : nn.LSTM
        LSTM taking concatenated [context, state] as input.
    head : nn.Sequential
        Projection head producing a scalar per step.
    """

    n_steps: int
    attn: Attention
    lstm: nn.LSTM
    head: nn.Sequential

    def __init__(self, hidden: int, n_layers: int, dropout: float, n_steps: int) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.attn = Attention(hidden)
        self.lstm = nn.LSTM(input_size=hidden * 2, hidden_size=hidden, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, enc_outputs: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode multi-step outputs.

        Parameters
        ----------
        enc_outputs : torch.Tensor
            Encoder outputs of shape (batch, seq_len, hidden).
        state : (torch.Tensor, torch.Tensor)
            Tuple of (h, c) where h,c have shape (n_layers, batch, hidden).

        Returns
        -------
        (torch.Tensor, (torch.Tensor, torch.Tensor))
            Tuple of (predictions (batch, n_steps, 1), final_state (h, c)).
        """
        preds = []
        h, c = state
        for _ in range(self.n_steps):
            h_prev = h[-1]
            context, _ = self.attn(h_prev, enc_outputs)
            dec_input = torch.cat([context, h_prev], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(dec_input, (h, c))
            step = self.head(out.squeeze(1))
            preds.append(step)
        stacked = torch.cat(preds, dim=1)
        return stacked.unsqueeze(-1), (h, c)

class MultiStepLSTMAttention(nn.Module):
    """
    Encoder–attention–decoder LSTM for multi-step forecasting.

    Parameters
    ----------
    n_feat : int
        Number of input features per time step.
    hidden : int, default 32
        Hidden size of the encoder/decoder.
    n_layers : int, default 3
        Number of LSTM layers in the encoder.
    dropout : float, default 0.2
        Dropout probability in the encoder LSTM.
    n_steps : int, default 1
        Number of forecast steps.

    Attributes
    ----------
    n_steps : int
        Number of forecast steps.
    encoder : nn.LSTM
        LSTM encoder.
    decoder : MultiStepLSTMDecoder
        Attention-equipped decoder.
    """

    n_steps: int
    encoder: nn.LSTM
    decoder: MultiStepLSTMDecoder

    def __init__(self, n_feat: int, hidden: int = 32, n_layers: int = 3, dropout: float = 0.2, n_steps: int = 1) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.encoder = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.decoder = MultiStepLSTMDecoder(hidden, n_layers, dropout, n_steps)

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
        enc_out, (h_n, c_n) = self.encoder(x)
        preds, _ = self.decoder(enc_out, (h_n, c_n))
        return preds

def build_lstm_attention(evaluator: Any) -> MultiStepLSTMAttention:
    """
    Convenience constructor for `MultiStepLSTMAttention`.

    Parameters
    ----------
    evaluator : Any
        Object providing `X_seq` (to infer n_feat) and `horizons` (to infer n_steps).

    Returns
    -------
    MultiStepLSTMAttention
        Instantiated attention LSTM model with preset hyperparameters.
    """
    n_feat = int(evaluator.X_seq.shape[2])
    n_steps = int(len(evaluator.horizons))
    return MultiStepLSTMAttention(n_feat, hidden=8, n_layers=2, dropout=0.2, n_steps=n_steps)
