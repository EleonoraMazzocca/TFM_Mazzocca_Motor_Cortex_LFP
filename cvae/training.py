"""Shared cVAE training utilities."""
from __future__ import annotations

import copy
from argparse import Namespace
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cvae.cvae_model import LFPCVAE, cvae_loss

_COND_PHASE_DIMS = slice(0, 3)
_COND_GRIP_DIMS = slice(3, 5)
_COND_HAND_DIMS = slice(5, 7)
_MMD_BANDWIDTHS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]


def apply_condition_dropout(
    c: torch.Tensor,
    p_single: float = 0.15,
    p_double: float = 0.04,
    p_all: float = 0.03,
) -> torch.Tensor:
    """Mask phase/grip/hand condition blocks for decoder-side regularization."""
    c_dec = c.clone()
    batch_size = c.shape[0]
    p_none = 1.0 - 3 * p_single - 3 * p_double - p_all
    if p_none < 0:
        raise ValueError("Condition dropout probabilities exceed 1.0")
    probs = torch.tensor(
        [p_none, p_single, p_single, p_single, p_double, p_double, p_double, p_all],
        dtype=torch.float32,
        device=c.device,
    )
    cases = torch.multinomial(probs, num_samples=batch_size, replacement=True)
    for i, case_tensor in enumerate(cases):
        case = int(case_tensor.item())
        if case in (1, 4, 5, 7):
            c_dec[i, _COND_PHASE_DIMS] = 0.0
        if case in (2, 4, 6, 7):
            c_dec[i, _COND_GRIP_DIMS] = 0.0
        if case in (3, 5, 6, 7):
            c_dec[i, _COND_HAND_DIMS] = 0.0
    return c_dec


def augment_embedding(
    x_clean: torch.Tensor,
    noise_scale: float = 0.1,
    amplitude_scale_range: tuple[float, float] = (0.85, 1.15),
    n_dropout_dims: int = 2,
) -> torch.Tensor:
    """Denoising augmentation for dense embedding vectors."""
    x_aug = x_clean.clone()
    batch_size, embed_dim = x_aug.shape
    for i in range(batch_size):
        if noise_scale != 0.0:
            std_i = x_aug[i].std(unbiased=False)
            x_aug[i] = x_aug[i] + torch.randn_like(x_aug[i]) * noise_scale * std_i
        if amplitude_scale_range != (1.0, 1.0):
            lo, hi = amplitude_scale_range
            scale = lo + (hi - lo) * torch.rand(1, device=x_aug.device).item()
            x_aug[i] = x_aug[i] * scale
        if n_dropout_dims > 0:
            drop_idx = torch.randperm(embed_dim, device=x_aug.device)[:n_dropout_dims]
            x_aug[i][drop_idx] = 0.0
    return x_aug


def mmd_loss_torch(z_enc: torch.Tensor, z_prior: torch.Tensor) -> torch.Tensor:
    """Differentiable multi-kernel MMD between encoded and prior samples."""
    total = z_enc.new_zeros(1)
    for bandwidth in _MMD_BANDWIDTHS:
        gamma = 1.0 / (2.0 * bandwidth ** 2)
        xx = torch.exp(-gamma * torch.cdist(z_enc, z_enc).pow(2)).mean()
        yy = torch.exp(-gamma * torch.cdist(z_prior, z_prior).pow(2)).mean()
        xy = torch.exp(-gamma * torch.cdist(z_enc, z_prior).pow(2)).mean()
        total = total + (xx + yy - 2.0 * xy)
    return total


def run_epoch(
    model: LFPCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
    device: torch.device,
    use_aug: bool = False,
    noise_scale: float = 0.1,
    amplitude_scale_range: tuple[float, float] = (0.85, 1.15),
    n_dropout_dims: int = 2,
    use_cond_dropout: bool = False,
    p_cond_single: float = 0.15,
    p_cond_double: float = 0.04,
    p_cond_all: float = 0.03,
    free_bits: float = 0.0,
    use_mmd_loss: bool = False,
    lambda_mmd: float = 10.0,
) -> tuple[float, float, float]:
    """Run one train/eval epoch. Third metric is KL in ELBO mode, MMD in MMD mode."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = total_recon = total_kl = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            x, c = batch[0].to(device), batch[1].to(device)
            x_clean = x
            x_input = augment_embedding(x_clean, noise_scale, amplitude_scale_range, n_dropout_dims) if training and use_aug else x_clean
            c_dec = apply_condition_dropout(c, p_cond_single, p_cond_double, p_cond_all) if training and use_cond_dropout else c

            if training and use_mmd_loss:
                mu, log_var = model.encode(x_input, c)
                std = torch.exp(0.5 * log_var)
                z = mu + std * torch.randn_like(std)
                x_recon = model.decode(z, c_dec)
                z_prior = torch.randn_like(z)
                recon = nn.functional.mse_loss(x_recon, x_clean, reduction="mean")
                kl = mmd_loss_torch(z, z_prior)
                loss = recon + lambda_mmd * kl
            else:
                x_recon, mu, log_var = model(x_input, c, c_dec=c_dec)
                loss, recon, kl = cvae_loss(x_recon, x_clean, mu, log_var, beta, free_bits=free_bits)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n = x.shape[0]
            total_loss += loss.item() * n
            total_recon += recon.item() * n
            total_kl += kl.item() * n

    n_samples = len(loader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples


def train_cvae(
    model: LFPCVAE,
    train_ds: Dataset,
    val_ds: Dataset,
    args: Namespace,
    device: torch.device,
    save_path: str | None = None,
    use_aug: bool = False,
    noise_scale: float = 0.1,
    amplitude_scale_range: tuple[float, float] = (0.85, 1.15),
    n_dropout_dims: int = 2,
    use_cond_dropout: bool = False,
    p_cond_single: float = 0.15,
    p_cond_double: float = 0.04,
    p_cond_all: float = 0.03,
    free_bits: float = 0.0,
    use_mmd_loss: bool = False,
    lambda_mmd: float = 10.0,
) -> dict:
    """Train a cVAE with KL annealing, optional MMD loss, and early stopping."""
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = {"train_loss": [], "train_recon": [], "train_kl": [], "val_loss": [], "val_recon": [], "val_kl": []}
    best_score = float("inf")
    best_val_loss = best_val_recon = best_val_kl = float("inf")
    best_state = None
    best_epoch = 0
    selection_metric = "val_recon" if use_mmd_loss else "val_loss"
    patience_ctr = 0
    no_early_stop = getattr(args, "no_early_stopping", False)
    n_epochs = 5 if args.dry_run else args.epochs
    latent_dim = getattr(args, "latent_dim", model.latent_dim)

    for epoch in range(n_epochs):
        beta = min(1.0, (epoch + 1) / max(args.beta_anneal_epochs, 1)) * args.beta_max
        tl, tr, tk = run_epoch(
            model,
            train_loader,
            optimizer,
            beta,
            device,
            use_aug=use_aug,
            noise_scale=noise_scale,
            amplitude_scale_range=amplitude_scale_range,
            n_dropout_dims=n_dropout_dims,
            use_cond_dropout=use_cond_dropout,
            p_cond_single=p_cond_single,
            p_cond_double=p_cond_double,
            p_cond_all=p_cond_all,
            free_bits=free_bits,
            use_mmd_loss=use_mmd_loss,
            lambda_mmd=lambda_mmd,
        )
        vl, vr, vk = run_epoch(model, val_loader, None, beta, device, free_bits=free_bits)
        history["train_loss"].append(tl)
        history["train_recon"].append(tr)
        history["train_kl"].append(tk)
        history["val_loss"].append(vl)
        history["val_recon"].append(vr)
        history["val_kl"].append(vk)
        third_label = "mmd" if use_mmd_loss else "kl"
        print(f"Epoch {epoch+1:03d}/{n_epochs} | recon={tr:.4f} {third_label}={tk:.4f} | val_recon={vr:.4f} val_kl={vk:.4f}")

        selection_score = vr if use_mmd_loss else vl
        if selection_score < best_score:
            best_score = selection_score
            best_val_loss = vl
            best_val_recon = vr
            best_val_kl = vk
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if not no_early_stop and patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}  (best {selection_metric}={best_score:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        torch.save(
            {
                "model_state": best_state or model.state_dict(),
                "history": history,
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_selection_metric": selection_metric,
                "best_selection_score": best_score,
                "best_val_loss": best_val_loss,
                "best_val_recon": best_val_recon,
                "best_val_kl_mean": best_val_kl,
                "best_val_kl_sum": best_val_kl * latent_dim,
                "use_mmd_loss": use_mmd_loss,
                "lambda_mmd": lambda_mmd,
            },
            save_path,
        )
        print(f"  Saved checkpoint: {save_path}")
    return history
