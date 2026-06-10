from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from data import N_AREAS

# MultiheadAttention gained `is_causal` in PyTorch 2.1; check once at import time.
_MHA_HAS_IS_CAUSAL = "is_causal" in inspect.signature(nn.MultiheadAttention.forward).parameters


class _AttnCapturingLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that saves per-head attention weights on every forward pass.

    Weights are stored as (batch, n_heads, seq_len, seq_len) in last_attn_weights.
    They are overwritten each forward call, so read them immediately after model(x).

    PyTorch ≥ 2.0 has a compiled fast path in TransformerEncoderLayer.forward() that
    bypasses _sa_block during eval mode (torch.backends.mha fast path).  We override
    forward() to disable that fast path for this layer so _sa_block is always called
    and attention weights are captured.  The fast path is restored after the call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        _mha = getattr(torch.backends, "mha", None)
        was_enabled = _mha.get_fastpath_enabled() if _mha is not None else False
        if _mha is not None:
            _mha.set_fastpath_enabled(False)
        try:
            return super().forward(
                src, src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        finally:
            if _mha is not None:
                _mha.set_fastpath_enabled(was_enabled)

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        mha_kwargs: dict = dict(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep per-head: (batch, n_heads, seq, seq)
        )
        if _MHA_HAS_IS_CAUSAL:
            mha_kwargs["is_causal"] = is_causal
        out, weights = self.self_attn(x, x, x, **mha_kwargs)
        self.last_attn_weights = weights.detach()
        return self.dropout1(out)


class LFPSpecialistTransformer(nn.Module):
    """
    Transformer that classifies grip, hand, and angle from a single movement phase.

    Two input modes:

    Area-avg mode (use_per_channel=False, default):
      One token per (area, bin) pair. For n_areas=4 and n_bins=10 → 40 tokens.
      Each token is the scalar area-average amplitude for one (area, bin), projected
      to d_model and summed with a learned area embedding and temporal position embedding.
      Token order is area-first: token k → (area k//n_bins, bin k%n_bins).
      Input: (batch, n_areas=4, n_bins)

    Per-channel mode (use_per_channel=True):
      One token per brain area (4 tokens). Each token is a vector of input_dim=96
      per-channel amplitudes (mean |x| over time, zero-padded to MAX_AREA_CHANNELS=96).
      No time_embedding. Input: (batch, n_areas=4, input_dim=96)

    Output: logits_grip (batch,2), logits_hand (batch,2), logits_angle (batch, n_angle_classes)
    """

    def __init__(
        self,
        n_areas: int = N_AREAS,
        n_bins: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.3,
        n_angle_classes: int = 4,
        use_per_channel: bool = False,
        input_dim: int = 1,
    ):
        super().__init__()
        self.n_areas = n_areas
        self.use_per_channel = use_per_channel

        if use_per_channel:
            # One token per area; each token is a vector of input_dim channel amplitudes.
            # n_bins is unused; no time_embedding.
            self.n_bins = 1          # sentinel so downstream reshape(n_areas, n_bins) still works
            self.n_tokens = n_areas
            self.input_proj = nn.Linear(input_dim, d_model)
            self.area_embedding = nn.Embedding(n_areas, d_model)
            self.register_buffer("area_idx", torch.arange(n_areas))
            self.register_buffer("bin_idx", torch.zeros(n_areas, dtype=torch.long))  # unused
        else:
            # One token per (area, bin) pair.
            self.n_bins = n_bins
            self.n_tokens = n_areas * n_bins
            self.input_proj = nn.Linear(1, d_model)
            self.area_embedding = nn.Embedding(n_areas, d_model)
            self.time_embedding = nn.Embedding(n_bins, d_model)
            area_idx = torch.arange(n_areas).repeat_interleave(n_bins)  # (n_tokens,)
            bin_idx = torch.arange(n_bins).repeat(n_areas)               # (n_tokens,)
            self.register_buffer("area_idx", area_idx)
            self.register_buffer("bin_idx", bin_idx)

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
        self.head_grip = nn.Linear(d_model, 2)
        self.head_hand = nn.Linear(d_model, 2)
        self.head_angle = nn.Linear(d_model, n_angle_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]

        if self.use_per_channel:
            # x: (batch, n_areas, input_dim)
            x = self.input_proj(x)                                     # (batch, n_areas, d_model)
            x = x + self.area_embedding(self.area_idx)[None, :, :]
        else:
            # x: (batch, n_areas, n_bins) — one scalar per token
            x = x.reshape(batch, self.n_tokens, 1)
            x = self.input_proj(x)                                     # (batch, n_tokens, d_model)
            x = x + self.area_embedding(self.area_idx)[None, :, :]
            x = x + self.time_embedding(self.bin_idx)[None, :, :]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x.mean(dim=1))                                   # (batch, d_model)
        return self.head_grip(x), self.head_hand(x), self.head_angle(x)

    def get_layer_attention_weights(self) -> list[torch.Tensor]:
        """Attention from the last forward pass, one tensor per layer.

        Returns list of (batch, n_heads, n_tokens, n_tokens).
        n_tokens = n_areas * n_bins.
        attn[b, h, i, j] = attention from token i (query) to token j (key).
        Token ordering: area-first → token k represents (area k//n_bins, bin k%n_bins).
        """
        return [
            layer.last_attn_weights
            for layer in self.layers
            if layer.last_attn_weights is not None
        ]
