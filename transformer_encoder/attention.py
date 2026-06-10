"""Attention-layer utilities shared by transformer encoders."""
from __future__ import annotations

import inspect

import torch
import torch.nn as nn

_MHA_HAS_IS_CAUSAL = "is_causal" in inspect.signature(nn.MultiheadAttention.forward).parameters


class AttnCapturingLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that saves per-head attention weights.

    Weights are stored as (batch, n_heads, seq_len, seq_len) in
    ``last_attn_weights`` and overwritten on each forward call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        mha_backend = getattr(torch.backends, "mha", None)
        was_enabled = mha_backend.get_fastpath_enabled() if mha_backend is not None else False
        if mha_backend is not None:
            mha_backend.set_fastpath_enabled(False)
        try:
            return super().forward(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        finally:
            if mha_backend is not None:
                mha_backend.set_fastpath_enabled(was_enabled)

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        kwargs = dict(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        if _MHA_HAS_IS_CAUSAL:
            kwargs["is_causal"] = is_causal
        out, weights = self.self_attn(x, x, x, **kwargs)
        self.last_attn_weights = weights.detach()
        return self.dropout1(out)
