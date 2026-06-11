"""Neural network for the joint phase/grip/hand transformer.

The model treats each LFP channel slot as one transformer token. For each sample,
features arrive as a small channel map:

    (batch, 4 brain areas, 96 channel slots per area, n_bands)

where n_bands is 1 for MU amplitude and 6 for broadband6. The model projects each
channel token into ``d_model`` dimensions, adds a learned brain-area embedding,
lets tokens communicate through self-attention, then averages the valid channel
tokens into one pooled embedding. That pooled embedding is the representation used
both by the three classification heads and by the downstream embedding cVAE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformer_encoder.attention import AttnCapturingLayer
from transformer_encoder.joint_embedding_data import AREA_IDX, N_AREAS, N_TOKENS, PADDING_MASK


class JointFactorTransformer(nn.Module):
    """Shared channel-token transformer for phase, grip, and hand classification.

    Important distinction:
    - ``n_heads`` is the number of transformer self-attention heads.
    - ``head_phase``, ``head_grip``, and ``head_hand`` are the three output
      classifiers.

    These two meanings of "head" are independent. We use 4 attention heads by
    default because ``d_model=64`` divides cleanly into 4 attention subspaces.
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

        # Per-channel feature vector -> transformer token embedding.
        # MU: one value per channel. broadband6: six frequency-band values per channel.
        self.input_proj = nn.Linear(n_bands, d_model)

        # Adds anatomical identity to each token. Without this, channel slot 10 in
        # one area and channel slot 10 in another area would look positionally similar.
        self.area_embedding = nn.Embedding(N_AREAS, d_model)

        # Transformer encoder layers. AttnCapturingLayer is a standard encoder
        # layer variant that remembers attention weights for later diagnostics.
        self.layers = nn.ModuleList([
            AttnCapturingLayer(
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

        # Three independent classifiers read the same pooled embedding.
        self.head_phase = nn.Linear(d_model, 3)
        self.head_grip = nn.Linear(d_model, 2)
        self.head_hand = nn.Linear(d_model, 2)

        # Buffers move with the model to CPU/GPU but are not trainable parameters.
        # area_idx: token -> brain area id, used for the learned area embedding.
        # pad_mask: True for padded/nonexistent channel slots, ignored by attention.
        self.register_buffer("area_idx", torch.tensor(AREA_IDX, dtype=torch.long))
        self.register_buffer("pad_mask", torch.tensor(PADDING_MASK, dtype=torch.bool))
        self.register_buffer("nonpad_mask", ~torch.tensor(PADDING_MASK, dtype=torch.bool))
        self.register_buffer("n_valid", (~torch.tensor(PADDING_MASK, dtype=torch.bool)).float().sum())

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return one pooled transformer embedding per sample.

        This method is the shared representation path:

        1. Flatten the area/channel grid into a sequence of channel tokens.
        2. Project each token's MU or broadband features into ``d_model`` dimensions.
        3. Add a learned area embedding so the model knows which brain area a token
           belongs to.
        4. Run self-attention across all valid channel tokens.
        5. Average only the real/non-padding tokens into one vector per sample.
        6. Layer-normalize the final vector.

        Shape summary:
        - input x: ``(batch, 4, 96, n_bands)``
        - token sequence: ``(batch, N_TOKENS, d_model)``
        - returned embedding: ``(batch, d_model)``

        The classifiers use this embedding directly, and ``run_embedding_cvae.py``
        saves/loads this same embedding space for generative modelling.
        """
        batch = x.shape[0]

        # Convert the 4x96 channel map into a token sequence. The order matches
        # AREA_IDX/PADDING_MASK from joint_embedding_data.py.
        tok = self.input_proj(x.reshape(batch, self.n_tokens, self.n_bands))

        # Give each token a learned anatomical label: M1, PMdL, PMdM, or PMv.
        tok = tok + self.area_embedding(self.area_idx)[None, :, :]

        # Attention should not read from padded channel slots. This does not
        # delete the padded tokens; it masks them inside each transformer layer.
        pad = self.pad_mask[None, :].expand(batch, -1)
        for layer in self.layers:
            tok = layer(tok, src_key_padding_mask=pad)

        # Pool tokens into a single sample-level representation. Padded channels
        # are zeroed before averaging, so they do not dilute the embedding.
        nonpad = self.nonpad_mask[None, :, None].float()
        pooled = (tok * nonpad).sum(dim=1) / self.n_valid
        return self.norm(pooled)

    def forward(self, x: torch.Tensor):
        """Predict phase, grip, and hand logits from one batch of LFP features.

        ``forward`` deliberately reuses ``extract_embedding``. That means the
        representation optimized by the supervised phase/grip/hand losses is
        exactly the representation exported later for the embedding cVAE.
        """
        emb = self.extract_embedding(x)
        return self.head_phase(emb), self.head_grip(emb), self.head_hand(emb)

    def get_layer_attention_weights(self) -> list[torch.Tensor]:
        """Return attention tensors captured during the most recent forward pass."""
        return [
            layer.last_attn_weights
            for layer in self.layers
            if layer.last_attn_weights is not None
        ]
