import math
import torch
import torch.nn as nn
from typing import Optional

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class TimeSeriesTransformerEncoder(nn.Module):
    """
    Transformer encoder for variable-length [B,T,C], with padding mask.
    - Projects C→d_model
    - Adds sinusoidal PE
    - nn.TransformerEncoder (PyTorch)
    - Pools over valid timesteps (masked mean)
    Returns:
      z_seq: [B,T,d_model] (sequence embeddings)
      z:     [B,d_model]   (pooled)
    """
    def __init__(self,
                 in_channels: int,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model, bias=True)
        self.pe = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B,T,D]
        mask: [B,T] boolean, True for valid tokens
        """
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        masked = x * mask.unsqueeze(-1)  # [B,T,D]
        return masked.sum(dim=1) / denom

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x:    [B,T,C]
        mask: [B,T] True for valid tokens, False for pad (matches your collate)
        PyTorch wants src_key_padding_mask with True = PAD; convert.
        """
        B, T, C = x.shape
        h = self.proj(x)                 # [B,T,D]
        h = self.pe(h)                   # add PE
        # src_key_padding_mask: [B,T] with True = PAD
        if mask is None:
            pad_mask = None
        else:
            pad_mask = ~mask             # invert (True=pad)

        z_seq = self.enc(h, src_key_padding_mask=pad_mask)   # [B,T,D]
        z_seq = self.norm(z_seq)

        if mask is None:
            z = z_seq.mean(dim=1)
        else:
            z = self._masked_mean(z_seq, mask)

        return z_seq, z
