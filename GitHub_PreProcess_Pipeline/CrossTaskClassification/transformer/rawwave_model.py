"""LFP Raw Waveform Transformer — 256 channels, each a 500-timepoint raw waveform token."""
from __future__ import annotations

import torch
import torch.nn as nn

from specialist_model import _AttnCapturingLayer
from rawwave_data import AREA_IDX_256, N_REAL_CHANNELS, N_TIMEPOINTS, N_AREAS


class LFPRawwaveTransformer(nn.Module):
    """
    256 tokens = 256 real LFP channels (no padding).
    Each token: 500-timepoint raw waveform projected to d_model=32.
    Area embedding tells transformer which brain area each channel belongs to.
    No padding mask — all tokens are real.
    """

    def __init__(
        self,
        n_channels: int = N_REAL_CHANNELS,   # 256
        n_timepoints: int = N_TIMEPOINTS,    # 500
        n_areas: int = N_AREAS,              # 4
        n_angle_classes: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_proj     = nn.Linear(n_timepoints, d_model)  # 500 → 32
        self.area_embedding = nn.Embedding(n_areas, d_model)
        self.layers = nn.ModuleList([
            _AttnCapturingLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.norm       = nn.LayerNorm(d_model)
        self.head_grip  = nn.Linear(d_model, 2)
        self.head_hand  = nn.Linear(d_model, 2)
        self.head_angle = nn.Linear(d_model, n_angle_classes)

        self.register_buffer("area_idx", torch.tensor(AREA_IDX_256, dtype=torch.long))

    def forward(
        self,
        x: torch.Tensor,  # (batch, 256, 500)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tok = self.input_proj(x)                                     # (batch, 256, d_model)
        tok = tok + self.area_embedding(self.area_idx)[None, :, :]
        for layer in self.layers:
            tok = layer(tok)                                         # no padding mask needed
        pooled = self.norm(tok.mean(dim=1))                          # (batch, d_model)
        return self.head_grip(pooled), self.head_hand(pooled), self.head_angle(pooled)

    def get_layer_attention_weights(self) -> list[torch.Tensor]:
        return [
            layer.last_attn_weights
            for layer in self.layers
            if layer.last_attn_weights is not None
        ]
