"""Conditional VAE (cVAE) model for LFP spectral amplitude generation.

Architecture:
  Encoder: concat(x, c) → MLP → (mu, log_var)
  Decoder: concat(z, c) → MLP → x_reconstructed

The condition vector c = [one_hot(phase,3), one_hot(grip,2), one_hot(hand,2)]
has dimension 7 and is injected into both encoder and decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(
    in_dim: int,
    hidden_dims: list[int],
    dropout: float,
) -> nn.Sequential:
    """Build a MLP with GELU activations and Dropout after every layer except the last.

    The last hidden layer has GELU but no Dropout (per spec).
    """
    layers: list[nn.Module] = []
    prev_dim = in_dim
    for i, h_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(nn.GELU())
        if i < len(hidden_dims) - 1:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim
    return nn.Sequential(*layers)


class LFPCVAE(nn.Module):
    """Conditional VAE conditioned on (phase, grip, hand).

    Args:
        input_dim     : feature dimensionality (256 spectral, 128000 raw)
        condition_dim : size of condition vector c (default 7)
        latent_dim    : size of latent code z (default 32)
        hidden_dims   : list of hidden layer sizes for encoder/decoder
        dropout       : dropout probability applied between hidden layers

    During training, z is sampled via reparameterization.
    During eval (model.eval()), generate() uses the prior N(0,I) to sample.
    model.reparameterize() is deterministic (returns mu) at eval time for
    reconstruction tasks — use generate() for generative sampling.
    """

    def __init__(
        self,
        input_dim:     int        = 256,
        condition_dim: int        = 7,
        latent_dim:    int        = 32,
        hidden_dims:   list[int]  = None,
        dropout:       float      = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_dim     = input_dim
        self.condition_dim = condition_dim
        self.latent_dim    = latent_dim

        enc_in = input_dim + condition_dim        # e.g. 256+7=263 for spectral

        # Encoder trunk: concat(x, c) → last hidden dim
        self.encoder = _build_mlp(enc_in, hidden_dims, dropout)

        # Variational heads on top of the encoder trunk
        self.mu_head      = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var_head = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder trunk: concat(z, c) → first hidden dim (reversed) → input_dim
        dec_hidden = list(reversed(hidden_dims))
        self.decoder = nn.Sequential(
            _build_mlp(latent_dim + condition_dim, dec_hidden, dropout),
            nn.Linear(dec_hidden[-1], input_dim),  # no activation on output
        )

    # ------------------------------------------------------------------
    # Forward components
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        """Encode (x, c) → (mu, log_var), each (B, latent_dim)."""
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.mu_head(h), self.log_var_head(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, exp(log_var)) during training; return mu during eval."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu   # deterministic at eval time for reconstruction tasks

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode (z, c) → reconstructed x, shape (B, input_dim)."""
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        c_dec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full cVAE forward pass.

        c     — condition fed to the encoder (always full).
        c_dec — condition fed to the decoder. If None, uses c unchanged.
                Pass a partially masked version during condition-dropout training.

        Returns (x_reconstructed, mu, log_var).
        """
        if c_dec is None:
            c_dec = c
        mu, log_var = self.encode(x, c)
        z           = self.reparameterize(mu, log_var)
        return self.decode(z, c_dec), mu, log_var

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        c:         torch.Tensor,
        n_samples: int = 1,
        device:    str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample n_samples from the prior N(0,I) and decode with condition c.

        c can be (7,) for a single condition (broadcast) or (n_samples, 7).
        Returns (n_samples, input_dim).
        Pass generator to isolate sampling randomness from the global RNG.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device, generator=generator)
            if c.dim() == 1:
                c = c.unsqueeze(0).expand(n_samples, -1)
            return self.decode(z, c.to(device))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cvae_loss(
    x_recon:   torch.Tensor,
    x:         torch.Tensor,
    mu:        torch.Tensor,
    log_var:   torch.Tensor,
    beta:      float = 1.0,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO = reconstruction_loss + beta * KL_loss.

    reconstruction_loss: mean MSE over batch and feature dimensions
    kl_loss:             mean raw KL divergence from N(0,I), always returned
                         unclamped for honest logging. When free_bits > 0, a
                         clamped copy is used for the loss only: dims below
                         the threshold receive no KL gradient, stopping the
                         prior from pulling mu further toward zero. This does
                         not actively push mu away from zero — recovery
                         depends on the reconstruction gradient through z.
    total:               reconstruction + beta * kl

    Returns (total, recon, kl) for logging.
    """
    recon = nn.functional.mse_loss(x_recon, x, reduction="mean")
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, latent_dim)
    kl_raw = kl_per_dim.mean()
    if free_bits > 0.0:
        kl_for_loss = torch.clamp(kl_per_dim, min=free_bits).mean()
    else:
        kl_for_loss = kl_raw
    total = recon + beta * kl_for_loss
    return total, recon, kl_raw
