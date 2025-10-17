from typing import Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention.

    Splits query/key/value across `num_heads`, performs scaled dot-product attention
    per head, then concatenates and projects back to `d_model`.

    Parameters
    ----------
    d_model : int
        Model (channel) dimension.
    num_heads : int
        Number of attention heads; must divide `d_model`.

    Attributes
    ----------
    d_model : int
        Model dimension.
    num_heads : int
        Number of heads.
    d_k : int
        Dimension per head (= d_model // num_heads).
    W_q, W_k, W_v : nn.Linear
        Projections for query, key, value.
    W_o : nn.Linear
        Output projection after head concatenation.
    """

    d_model: int
    num_heads: int
    d_k: int
    W_q: nn.Linear
    W_k: nn.Linear
    W_v: nn.Linear
    W_o: nn.Linear

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute scaled dot-product attention for pre-split heads."""
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, T, d_model) -> (B, H, T, d_k)."""
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, H, T, d_k) -> (B, T, d_model)."""
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        Q, K, V : torch.Tensor
            Tensors of shape (B, T, d_model).
        mask : torch.Tensor or None
            Optional attention mask broadcastable to (B, H, T, T).

        Returns
        -------
        torch.Tensor
            Output of shape (B, T, d_model).
        """
        Qh = self.split_heads(self.W_q(Q)); Kh = self.split_heads(self.W_k(K)); Vh = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Qh, Kh, Vh, mask)
        return self.W_o(self.combine_heads(attn_output))

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feedforward network.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_ff : int
        Hidden layer dimension.

    Attributes
    ----------
    fc1, fc2 : nn.Linear
        Linear layers.
    relu : nn.ReLU
        Activation.
    """

    fc1: nn.Linear
    fc2: nn.Linear
    relu: nn.ReLU

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN: Linear -> ReLU -> Linear."""
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (added to inputs).

    Parameters
    ----------
    d_model : int
        Model dimension.
    max_seq_length : int
        Maximum sequence length supported.

    Attributes
    ----------
    pe : torch.Buffer
        Positional encodings of shape (1, max_seq_length, d_model).
    """

    def __init__(self, d_model: int, max_seq_length: int) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the first `x.size(1)` positions."""
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    """
    Minimal encoder layer: self-attention + feedforward with residual and norm.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    d_ff : int
        Feedforward hidden size.
    dropout : float
        Dropout prob (applied before residual).

    Notes
    -----
    `target_emb` is preserved for API compatibility with the original code.
    """

    self_attn: MultiHeadAttention
    feed_forward: PositionWiseFeedForward
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    dropout: nn.Dropout
    target_emb: nn.Linear

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.target_emb = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply self-attn then FFN with residual connections and layer norms."""
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))

class DecoderLayer(nn.Module):
    """
    Minimal decoder layer: masked self-attn + cross-attn + feedforward.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    d_ff : int
        Feedforward hidden size.
    dropout : float
        Dropout prob (applied before residual).
    """

    self_attn: MultiHeadAttention
    cross_attn: MultiHeadAttention
    feed_forward: PositionWiseFeedForward
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    norm3: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply masked self-attn, cross-attn with encoder output, then FFN."""
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))

class MultiStepTransformer(nn.Module):
    """
    Lightweight seq2seq Transformer for multi-step forecasting.

    Encoder/decoder are constructed from bare multi-head attention blocks
    with position-wise feedforward and layer norms, followed by an
    autoregressive decoder that emits `n_steps` scalar predictions.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep.
    d_model : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of encoder/decoder attention blocks.
    d_ff : int
        Feedforward hidden size.
    max_seq_length : int
        Maximum sequence length for positional encoding.
    dropout : float
        Dropout probability.
    n_steps : int
        Number of forecast steps.

    Attributes
    ----------
    n_steps : int
        Number of forecast steps.
    input_proj : nn.Linear
        Projects inputs to model dimension.
    pos_enc : PositionalEncoding
        Sinusoidal positional encoding module.
    encoder_layers : nn.ModuleList
        List of encoder attention blocks.
    ff_enc : PositionWiseFeedForward
        Encoder feedforward.
    norm_enc : nn.LayerNorm
        Encoder normalization.
    decoder_layers : nn.ModuleList
        List of decoder attention blocks.
    ff_dec : PositionWiseFeedForward
        Decoder feedforward.
    norm_dec : nn.LayerNorm
        Decoder normalization.
    output_head : nn.Linear
        Final linear head mapping to a scalar.
    target_emb : nn.Linear
        Embeds previous output for AR decoding.
    """

    n_steps: int
    input_proj: nn.Linear
    pos_enc: PositionalEncoding
    encoder_layers: nn.ModuleList
    ff_enc: PositionWiseFeedForward
    norm_enc: nn.LayerNorm
    decoder_layers: nn.ModuleList
    ff_dec: PositionWiseFeedForward
    norm_dec: nn.LayerNorm
    output_head: nn.Linear
    target_emb: nn.Linear

    def __init__(self, input_dim: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_seq_length: int, dropout: float, n_steps: int) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])
        self.ff_enc = PositionWiseFeedForward(d_model, d_ff)
        self.norm_enc = nn.LayerNorm(d_model)
        self.decoder_layers = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])
        self.ff_dec = PositionWiseFeedForward(d_model, d_ff)
        self.norm_dec = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)
        self.target_emb = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with autoregressive decoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch, n_steps, 1).
        """
        batch, _, _ = x.size()
        h = self.pos_enc(self.input_proj(x))
        for attn in self.encoder_layers:
            h = attn(h, h, h)
            h = self.norm_enc(h + self.ff_enc(h))

        preds: List[torch.Tensor] = []
        dec_input = torch.zeros(batch, 1, self.input_proj.out_features, device=x.device)
        state = h
        for _ in range(self.n_steps):
            d = self.pos_enc(dec_input)
            for attn in self.decoder_layers:
                d = attn(d, d, d)
                d = attn(d, state, state)
                d = self.norm_dec(d + self.ff_dec(d))
            out_step = self.output_head(d[:, -1, :]).unsqueeze(1)
            preds.append(out_step)
            dec_input = torch.cat([dec_input, self.target_emb(out_step)], dim=1)
        return torch.cat(preds, dim=1)

def build_transformer(evaluator: Any) -> MultiStepTransformer:
    """
    Convenience constructor for `MultiStepTransformer` using evaluator metadata.

    Parameters
    ----------
    evaluator : Any
        Object with `X_seq` (to infer input_dim and max length) and `horizons` (to infer n_steps).

    Returns
    -------
    MultiStepTransformer
        Instantiated model with reasonable defaults.
    """
    n_feat = int(evaluator.X_seq.shape[2])
    n_steps = int(len(evaluator.horizons))
    max_len = int(evaluator.X_seq.shape[1] + n_steps)
    return MultiStepTransformer(input_dim=n_feat, d_model=64, num_heads=4, num_layers=2, d_ff=128, max_seq_length=max_len, dropout=0.1, n_steps=n_steps)
