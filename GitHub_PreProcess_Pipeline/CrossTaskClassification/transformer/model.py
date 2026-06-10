from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import AREA_FEATURE_DIM, N_AREAS, N_PHASES


class LFPTransformerClassifier(nn.Module):
    """
    Transformer that classifies grip, hand, and angle from 12 tokens:
    one per (phase × brain area).

    Input shape:  (batch, 3, 4, AREA_FEATURE_DIM)
    Output:       three logit tensors (grip×2, hand×2, angle×4)
    """

    def __init__(
        self,
        n_phases: int = N_PHASES,
        n_areas: int = N_AREAS,
        area_feature_dim: int = AREA_FEATURE_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_phases = n_phases
        self.n_areas = n_areas

        self.input_proj = nn.Linear(area_feature_dim, d_model)
        self.phase_embedding = nn.Embedding(n_phases, d_model)
        self.area_embedding = nn.Embedding(n_areas, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head_grip = nn.Linear(d_model, 2)
        self.head_hand = nn.Linear(d_model, 2)
        self.head_angle = nn.Linear(d_model, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, n_phases, n_areas, area_feature_dim)
        batch = x.shape[0]

        x = self.input_proj(x)  # (batch, n_phases, n_areas, d_model)

        phase_idx = torch.arange(self.n_phases, device=x.device)
        area_idx = torch.arange(self.n_areas, device=x.device)
        x = (
            x
            + self.phase_embedding(phase_idx)[None, :, None, :]
            + self.area_embedding(area_idx)[None, None, :, :]
        )

        x = x.reshape(batch, self.n_phases * self.n_areas, -1)  # (batch, 12, d_model)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))  # (batch, d_model)
        return self.head_grip(x), self.head_hand(x), self.head_angle(x)


class LFPInstructionTransformer(nn.Module):
    """
    LFPTransformerClassifier extended with late concat-fusion of an instruction vector.

    The LFP encoder is identical to the baseline. After mean-pooling the 12
    (phase × area) tokens, the instruction is projected to instruction_proj_dim and
    concatenated with the pooled LFP representation before the classification heads.
    At test time the instruction is always the zero vector, so the model must decode
    from LFP alone — the instruction can only assist during training.

    Args:
        instruction_dim:      must match the encoding used (8=onehot, 9=bow, 384=minilm)
        instruction_proj_dim: project instruction to this dim before concat (default 32)

    Input:
        x:     (batch, n_phases, n_areas, area_feature_dim)
        instr: (batch, instruction_dim) — zeroed by BalancedInstructionDataset at test time

    Output:
        three logit tensors (grip×2, hand×2, angle×4)

    Internal flow:
        pooled = mean_pool(encoder(tokens))          # (batch, d_model)
        instr  = relu(instruction_proj(instruction)) # (batch, instruction_proj_dim)
        fused  = cat([pooled, instr], dim=1)         # (batch, d_model + instruction_proj_dim)
        → head_grip, head_hand, head_angle
    """

    def __init__(
        self,
        n_phases: int = N_PHASES,
        n_areas: int = N_AREAS,
        area_feature_dim: int = AREA_FEATURE_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        instruction_dim: int = 8,
        instruction_proj_dim: int = 32,
    ):
        super().__init__()
        self.n_phases = n_phases
        self.n_areas = n_areas

        # LFP pathway (identical to baseline)
        self.input_proj = nn.Linear(area_feature_dim, d_model)
        self.phase_embedding = nn.Embedding(n_phases, d_model)
        self.area_embedding = nn.Embedding(n_areas, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Instruction pathway: small projection to avoid over-parameterizing
        self.instruction_proj = nn.Linear(instruction_dim, instruction_proj_dim)

        # Heads receive the fused representation
        fused_dim = d_model + instruction_proj_dim
        self.head_grip = nn.Linear(fused_dim, 2)
        self.head_hand = nn.Linear(fused_dim, 2)
        self.head_angle = nn.Linear(fused_dim, 4)

    def forward(
        self,
        x: torch.Tensor,
        instr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x:     (batch, n_phases, n_areas, area_feature_dim)
        # instr: (batch, instruction_dim)
        batch = x.shape[0]

        # LFP pathway
        x = self.input_proj(x)
        phase_idx = torch.arange(self.n_phases, device=x.device)
        area_idx = torch.arange(self.n_areas, device=x.device)
        x = (
            x
            + self.phase_embedding(phase_idx)[None, :, None, :]
            + self.area_embedding(area_idx)[None, None, :, :]
        )
        x = x.reshape(batch, self.n_phases * self.n_areas, -1)  # (batch, 12, d_model)
        x = self.encoder(x)
        pooled = self.norm(x.mean(dim=1))  # (batch, d_model)

        # Instruction pathway: project then ReLU
        instr_out = F.relu(self.instruction_proj(instr))  # (batch, instruction_proj_dim)

        # Concat fusion — heads take the full fused vector
        fused = torch.cat([pooled, instr_out], dim=1)  # (batch, d_model + instruction_proj_dim)

        return self.head_grip(fused), self.head_hand(fused), self.head_angle(fused)
