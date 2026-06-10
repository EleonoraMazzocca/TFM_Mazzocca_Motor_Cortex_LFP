"""Joint phase/grip/hand channel-token transformer."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformer_encoder.specialist_model import _AttnCapturingLayer
from transformer_encoder.joint_embedding_data import AREA_IDX, N_AREAS, N_TOKENS, PADDING_MASK


class JointFactorTransformer(nn.Module):
    """Shared channel-token transformer for phase, grip, and hand classification.

    Input shape: (batch, 4, 96, n_bands).  For MU, n_bands=1.  For broadband6,
    n_bands=6.  Tokens are channel slots, with padded M1/PMdL slots masked.
    """

    def __init__(
        self,
        n_bands: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.d_model = d_model
        self.n_tokens = N_TOKENS

        self.input_proj = nn.Linear(n_bands, d_model)
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
        self.norm = nn.LayerNorm(d_model)
        self.head_phase = nn.Linear(d_model, 3)
        self.head_grip = nn.Linear(d_model, 2)
        self.head_hand = nn.Linear(d_model, 2)

        self.register_buffer("area_idx", torch.tensor(AREA_IDX, dtype=torch.long))
        self.register_buffer("pad_mask", torch.tensor(PADDING_MASK, dtype=torch.bool))
        self.register_buffer("nonpad_mask", ~torch.tensor(PADDING_MASK, dtype=torch.bool))
        self.register_buffer("n_valid", (~torch.tensor(PADDING_MASK, dtype=torch.bool)).float().sum())

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tok = self.input_proj(x.reshape(batch, self.n_tokens, self.n_bands))
        tok = tok + self.area_embedding(self.area_idx)[None, :, :]
        pad = self.pad_mask[None, :].expand(batch, -1)
        for layer in self.layers:
            tok = layer(tok, src_key_padding_mask=pad)
        nonpad = self.nonpad_mask[None, :, None].float()
        pooled = (tok * nonpad).sum(dim=1) / self.n_valid
        return self.norm(pooled)

    def forward(self, x: torch.Tensor):
        emb = self.extract_embedding(x)
        return self.head_phase(emb), self.head_grip(emb), self.head_hand(emb)

    def get_layer_attention_weights(self) -> list[torch.Tensor]:
        return [
            layer.last_attn_weights
            for layer in self.layers
            if layer.last_attn_weights is not None
        ]

