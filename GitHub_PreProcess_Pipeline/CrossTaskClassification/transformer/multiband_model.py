"""LFP Multiband Transformer — attends over 384 per-channel spectral fingerprints."""
from __future__ import annotations

import torch
import torch.nn as nn

from specialist_model import _AttnCapturingLayer
from multiband_data import (
    AREA_IDX, MAX_AREA_CHANNELS, N_AREAS, N_BANDS, N_TOKENS, PADDING_MASK,
)


class LFPMultibandTransformer(nn.Module):
    """Transformer for multi-band LFP classification.

    Input : (batch, N_AREAS, MAX_AREA_CHANNELS, N_BANDS) = (batch, 4, 96, 6)
    Tokens: 384 = 4 areas × 96 channels (padded M1/PMdL slots are masked out)
    Output: logits_grip (batch,2), logits_hand (batch,2), logits_angle (batch, n_angle_classes)
    """

    def __init__(
        self,
        n_angle_classes: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.n_angle_classes = n_angle_classes
        self.d_model = d_model
        self.n_tokens = N_TOKENS  # 384

        self.input_proj    = nn.Linear(N_BANDS, d_model)
        self.area_embedding = nn.Embedding(N_AREAS, d_model)

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

        self.register_buffer("area_idx",   torch.tensor(AREA_IDX,     dtype=torch.long))
        self.register_buffer("pad_mask",   torch.tensor(PADDING_MASK, dtype=torch.bool))
        self.register_buffer("nonpad_mask", ~torch.tensor(PADDING_MASK, dtype=torch.bool))
        self.register_buffer("n_valid",
                             (~torch.tensor(PADDING_MASK, dtype=torch.bool)).float().sum())

    def forward(
        self,
        x: torch.Tensor,  # (batch, N_AREAS, MAX_AREA_CHANNELS, N_BANDS)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]

        tok = self.input_proj(x.reshape(batch, self.n_tokens, N_BANDS))   # (batch, 384, d_model)
        tok = tok + self.area_embedding(self.area_idx)[None, :, :]

        pad = self.pad_mask[None, :].expand(batch, -1)   # (batch, 384)
        for layer in self.layers:
            tok = layer(tok, src_key_padding_mask=pad)

        # Mean pool over non-padded tokens only
        nonpad = self.nonpad_mask[None, :, None].float()  # (1, 384, 1)
        pooled = (tok * nonpad).sum(dim=1) / self.n_valid  # (batch, d_model)

        out = self.norm(pooled)
        return self.head_grip(out), self.head_hand(out), self.head_angle(out)

    def get_layer_attention_weights(self) -> list[torch.Tensor]:
        return [
            layer.last_attn_weights
            for layer in self.layers
            if layer.last_attn_weights is not None
        ]
