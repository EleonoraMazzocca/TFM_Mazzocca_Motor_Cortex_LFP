"""Evaluation helpers for embedding-space cVAE experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader

from cvae.conditioning.onehot import make_condition_vector
from cvae.conditioning.sentence import lookup_condition
from cvae.metrics import compute_mmd
from transformer_encoder.joint_embedding_data import GRIP_TO_ID, HAND_TO_ID, PHASE_NAMES

ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}


def reconstruct_seen(model, val_ds, device, batch_size: int = 128) -> dict:
    """Measure how well the cVAE reconstructs seen validation embeddings."""
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    xs, xrs, gs, hs, ps = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, c, yg, yh, ya, yp in loader:
            xr, _, _ = model(x.to(device), c.to(device))
            xs.append(x.numpy())
            xrs.append(xr.cpu().numpy())
            gs.append(yg.numpy())
            hs.append(yh.numpy())
            ps.append(yp.numpy())
    x = np.concatenate(xs)
    xr = np.concatenate(xrs)
    g = np.concatenate(gs)
    h = np.concatenate(hs)
    p = np.concatenate(ps)

    by_combo = {}
    for ph in range(len(PHASE_NAMES)):
        for gi in (0, 1):
            for hi in (0, 1):
                mask = (p == ph) & (g == gi) & (h == hi)
                if not mask.any():
                    continue
                label = f"{PHASE_NAMES[ph]} + {ID_TO_GRIP[gi]} + {ID_TO_HAND[hi]}"
                mse = float(np.mean((x[mask] - xr[mask]) ** 2))
                r = float(np.corrcoef(x[mask].reshape(-1), xr[mask].reshape(-1))[0, 1])
                by_combo[label] = {"mse": mse, "pearsonr": r}
    return {
        "by_combo": by_combo,
        "mse_mean": float(np.mean([v["mse"] for v in by_combo.values()])),
        "pearsonr_mean": float(np.mean([v["pearsonr"] for v in by_combo.values()])),
    }


def evaluate_generation(
    model,
    transformer_model,
    train_payload: dict,
    heldout_payload: dict,
    stats: dict,
    condition: np.ndarray,
    args,
    device,
) -> tuple[dict, dict]:
    """Evaluate held-out generation quality and return plot-ready artifacts."""
    x_real = (heldout_payload["embeddings"].astype(np.float32) - stats["mu"]) / stats["sigma"]
    c = torch.tensor(condition, dtype=torch.float32)
    x_gen = model.generate(c, n_samples=500, device=device).cpu().numpy()

    head_accuracy = None
    preds_gen = None
    preds_real = None
    head_targets = None
    head_class_names = {"phase": PHASE_NAMES, "grip": ["power", "precision"], "hand": ["left", "right"]}
    if transformer_model is not None:
        transformer_model.eval()
        mu_np = stats["mu"].astype(np.float32)
        sigma_np = stats["sigma"].astype(np.float32)
        x_gen_denorm = torch.tensor(x_gen * sigma_np + mu_np, dtype=torch.float32).to(device)
        x_real_denorm = torch.tensor(heldout_payload["embeddings"].astype(np.float32), dtype=torch.float32).to(device)
        target_phase_idx = PHASE_NAMES.index(args.heldout_phase)
        target_grip_id = GRIP_TO_ID[args.heldout_grip]
        target_hand_id = HAND_TO_ID[args.heldout_hand]
        head_targets = {"phase": target_phase_idx, "grip": target_grip_id, "hand": target_hand_id}
        with torch.no_grad():
            preds_gen = {
                "phase": transformer_model.head_phase(x_gen_denorm).argmax(1).cpu().numpy(),
                "grip": transformer_model.head_grip(x_gen_denorm).argmax(1).cpu().numpy(),
                "hand": transformer_model.head_hand(x_gen_denorm).argmax(1).cpu().numpy(),
            }
            preds_real = {
                "phase": transformer_model.head_phase(x_real_denorm).argmax(1).cpu().numpy(),
                "grip": transformer_model.head_grip(x_real_denorm).argmax(1).cpu().numpy(),
                "hand": transformer_model.head_hand(x_real_denorm).argmax(1).cpu().numpy(),
            }
        head_accuracy = {
            "generated": {f: float((preds_gen[f] == head_targets[f]).mean()) for f in ("phase", "grip", "hand")},
            "real_heldout": {f: float((preds_real[f] == head_targets[f]).mean()) for f in ("phase", "grip", "hand")},
        }

    p_vals = []
    for j in range(x_real.shape[1]):
        _, p = ttest_ind(x_real[:, j], x_gen[:, j], equal_var=False)
        p_vals.append(float(p))
    frac_ns = float(np.mean(np.asarray(p_vals) > 0.05))
    mmd_gen = compute_mmd(x_gen, x_real)
    half = len(x_real) // 2
    mmd_base = compute_mmd(x_real[:half], x_real[half:]) if half > 1 else float("nan")
    mmd_ratio = float(mmd_gen / max(mmd_base, 1e-10)) if not np.isnan(mmd_base) else float("nan")

    x_train = (train_payload["embeddings"].astype(np.float32) - stats["mu"]) / stats["sigma"]
    target_phase = PHASE_NAMES.index(args.heldout_phase)
    target_grip = GRIP_TO_ID[args.heldout_grip]
    target_hand = HAND_TO_ID[args.heldout_hand]
    target_combo = target_grip * 6 + target_hand * 3 + target_phase

    train_combo = (
        train_payload["y_grip"].astype(np.int64) * 6
        + train_payload["y_hand"].astype(np.int64) * 3
        + train_payload["y_phase"].astype(np.int64)
    )

    target_centroid = x_real.mean(axis=0)
    gen_centroid = x_gen.mean(axis=0)
    centroid_dist = float(np.linalg.norm(gen_centroid - target_centroid))
    centroids = {}
    for combo in sorted(np.unique(train_combo)):
        centroids[int(combo)] = x_train[train_combo == combo].mean(axis=0)
    centroids[int(target_combo)] = target_centroid
    centroid_keys = np.array(sorted(centroids))
    centroid_mat = np.stack([centroids[int(k)] for k in centroid_keys], axis=0)
    other_mask = centroid_keys != target_combo
    other_keys = centroid_keys[other_mask]
    target_to_other = np.linalg.norm(centroid_mat[other_mask] - target_centroid[None, :], axis=1)
    target_to_other_mean = float(np.mean(target_to_other)) if len(target_to_other) else float("nan")
    target_to_other_min = float(np.min(target_to_other)) if len(target_to_other) else float("nan")
    relative_centroid_distance_mean = float(centroid_dist / max(target_to_other_mean, 1e-10))
    relative_centroid_distance_min = float(centroid_dist / max(target_to_other_min, 1e-10))
    centroid_distance_table = []
    for combo, dist in sorted(zip(other_keys, target_to_other), key=lambda x: float(x[1])):
        centroid_distance_table.append({
            "combo_id": int(combo),
            "combo": f"{PHASE_NAMES[int(combo) % 3]}+{ID_TO_GRIP[int(combo) // 6]}+{ID_TO_HAND[(int(combo) // 3) % 2]}",
            "distance_to_target": float(dist),
            "closer_than_generated": bool(float(dist) < centroid_dist),
            "distance_minus_generated": float(dist - centroid_dist),
        })
    closer_than_generated = [row for row in centroid_distance_table if row["closer_than_generated"]]
    dists = np.linalg.norm(x_gen[:, None, :] - centroid_mat[None, :, :], axis=2)
    nearest = centroid_keys[np.argmin(dists, axis=1)]
    nearest_target_rate = float((nearest == target_combo).mean())
    mean_nearest_margin = float(np.mean(np.partition(dists, kth=1, axis=1)[:, 1] - np.min(dists, axis=1)))

    metrics = {
        "frac_dims_p_gt_005": frac_ns,
        "mmd_generated": float(mmd_gen),
        "mmd_baseline": float(mmd_base),
        "mmd_ratio": mmd_ratio,
        "centroid_distance": centroid_dist,
        "target_to_other_centroid_mean": target_to_other_mean,
        "target_to_other_centroid_min": target_to_other_min,
        "relative_centroid_distance_mean": relative_centroid_distance_mean,
        "relative_centroid_distance_min": relative_centroid_distance_min,
        "nearest_centroid_target_rate": nearest_target_rate,
        "nearest_centroid_mean_margin": mean_nearest_margin,
        "centroid_distance_table": centroid_distance_table,
        "centroids_closer_than_generated": closer_than_generated,
        "head_accuracy": head_accuracy,
    }
    plot_payload = {
        "x_train": x_train,
        "x_real": x_real,
        "x_gen": x_gen,
        "train_payload": train_payload,
        "target_phase": target_phase,
        "target_grip": target_grip,
        "target_hand": target_hand,
        "target_combo": target_combo,
        "centroid_keys": centroid_keys,
        "centroid_mat": centroid_mat,
        "gen_centroid": gen_centroid,
        "target_centroid": target_centroid,
        "nearest_target_rate": nearest_target_rate,
        "centroid_dist": centroid_dist,
        "target_to_other_min": target_to_other_min,
        "target_to_other_mean": target_to_other_mean,
        "relative_centroid_distance_mean": relative_centroid_distance_mean,
        "relative_centroid_distance_min": relative_centroid_distance_min,
        "centroid_distance_table": centroid_distance_table,
        "head_accuracy": head_accuracy,
        "preds_gen": preds_gen,
        "preds_real": preds_real,
        "head_targets": head_targets,
        "head_class_names": head_class_names,
    }
    return metrics, plot_payload


def compute_collapse_diagnostics(
    model,
    val_ds,
    heldout_ds,
    args,
    device,
    out_dir: Path,
    val_indices: np.ndarray,
    generation_seed: int = 0,
) -> dict:
    """Encode val set and compute posterior-collapse diagnostics."""
    model.eval()

    loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    mu_list, lv_list, grip_l, hand_l, phase_l = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x_b, c_b = batch[0].to(device), batch[1].to(device)
            mu_b, lv_b = model.encode(x_b, c_b)
            mu_list.append(mu_b.cpu().numpy())
            lv_list.append(lv_b.cpu().numpy())
            grip_l.append(batch[2].numpy())
            hand_l.append(batch[3].numpy())
            phase_l.append(batch[5].numpy())

    mu_all = np.concatenate(mu_list, axis=0)
    lv_all = np.concatenate(lv_list, axis=0)
    grip_v = np.concatenate(grip_l)
    hand_v = np.concatenate(hand_l)
    phase_v = np.concatenate(phase_l)
    val_combo_labels = (grip_v * 6 + hand_v * 3 + phase_v).astype(np.int64)

    std_mu = mu_all.std(axis=0, ddof=0)
    mean_sigma = np.exp(0.5 * lv_all).mean(axis=0)
    second_moment = (mu_all ** 2 + np.exp(lv_all)).mean(axis=0)

    print(f"\n  Collapse diagnostics (latent_dim={mu_all.shape[1]}):")
    print(f"    std(mu) — mean={std_mu.mean():.4f}  min={std_mu.min():.4f}  max={std_mu.max():.4f}")
    print(f"    mean(sigma) — mean={mean_sigma.mean():.4f}")
    print(f"    second_moment — mean={second_moment.mean():.4f}  (target ~1.0)")

    seen_mmd_records = {}
    seen_bandwidths = {}
    unique_combos = np.unique(val_combo_labels)
    for combo in unique_combos:
        mask = val_combo_labels == combo
        x_real_combo = np.stack([val_ds[int(i)][0].numpy() for i in np.where(mask)[0]])
        if len(x_real_combo) < 2:
            continue
        from sklearn.metrics.pairwise import euclidean_distances as _edist

        dists_combo = _edist(x_real_combo, x_real_combo)
        np.fill_diagonal(dists_combo, np.nan)
        bw = float(np.nanmedian(dists_combo))
        bw = max(bw, 1e-6)

        phase_id = int(combo % 3)
        hand_id = int((combo // 3) % 2)
        grip_id = int(combo // 6)
        _ct = getattr(args, "condition_table", None)
        _ck = getattr(args, "condition_key_order", None)
        cond_vec = (
            lookup_condition(phase_id, grip_id, hand_id, _ct, _ck)
            if _ct is not None else make_condition_vector(phase_id, grip_id, hand_id)
        )
        c_tensor = torch.tensor(cond_vec, dtype=torch.float32)
        gen_rng = torch.Generator(device=device).manual_seed(generation_seed)
        x_gen_combo = model.generate(c_tensor, n_samples=500, device=device, generator=gen_rng).cpu().numpy()

        mmd_val = compute_mmd(x_gen_combo, x_real_combo, bandwidth=bw)
        ph_name = PHASE_NAMES[phase_id]
        grip_name = ID_TO_GRIP[grip_id]
        hand_name = ID_TO_HAND[hand_id]
        label = f"{ph_name}+{grip_name}+{hand_name}"
        seen_mmd_records[label] = {"mmd": mmd_val, "bandwidth": bw, "n_real": int(x_real_combo.shape[0]), "n_generated": 500}
        seen_bandwidths[label] = bw
        print(f"    seen MMD [{label}]: {mmd_val:.5f}  (bw={bw:.4f})")

    held_loader = DataLoader(heldout_ds, batch_size=256, shuffle=False)
    x_held_l = []
    with torch.no_grad():
        for batch in held_loader:
            x_held_l.append(batch[0].numpy())
    x_held = np.concatenate(x_held_l, axis=0)

    from sklearn.metrics.pairwise import euclidean_distances as _edist2

    dists_held = _edist2(x_held, x_held)
    np.fill_diagonal(dists_held, np.nan)
    bw_held = float(np.nanmedian(dists_held))
    bw_held = max(bw_held, 1e-6)

    heldout_phase_idx = PHASE_NAMES.index(args.heldout_phase)
    heldout_grip_id = GRIP_TO_ID[args.heldout_grip]
    heldout_hand_id = HAND_TO_ID[args.heldout_hand]
    _ct = getattr(args, "condition_table", None)
    _ck = getattr(args, "condition_key_order", None)
    cond_held = (
        lookup_condition(heldout_phase_idx, heldout_grip_id, heldout_hand_id, _ct, _ck)
        if _ct is not None else make_condition_vector(heldout_phase_idx, heldout_grip_id, heldout_hand_id)
    )
    c_held_t = torch.tensor(cond_held, dtype=torch.float32)
    gen_rng_held = torch.Generator(device=device).manual_seed(generation_seed)
    x_gen_held = model.generate(c_held_t, n_samples=500, device=device, generator=gen_rng_held).cpu().numpy()

    mmd_held = compute_mmd(x_gen_held, x_held, bandwidth=bw_held)
    print(f"    held-out MMD: {mmd_held:.5f}  (bw={bw_held:.4f}  n_real={len(x_held)})")

    np.savez_compressed(
        out_dir / "collapse_diagnostics.npz",
        std_mu=std_mu,
        mean_sigma=mean_sigma,
        second_moment=second_moment,
        mu_all=mu_all,
        val_combo_labels=val_combo_labels,
        val_indices=val_indices,
        mmd_seen=np.bytes_(json.dumps(seen_mmd_records)),
        mmd_seen_bandwidths=np.bytes_(json.dumps(seen_bandwidths)),
        mmd_heldout=np.array(mmd_held),
        mmd_heldout_bandwidth=np.array(bw_held),
        generation_seed=np.array(generation_seed),
        latent_dim=np.array(args.latent_dim),
        split_seed=np.array(args.split_seed),
        heldout_combo=np.array([args.heldout_phase, args.heldout_grip, args.heldout_hand]),
    )
    print(f"  Saved collapse_diagnostics.npz to {out_dir}")

    return {
        "std_mu_mean": float(std_mu.mean()),
        "mean_sigma_mean": float(mean_sigma.mean()),
        "second_moment_mean": float(second_moment.mean()),
        "mmd_heldout": float(mmd_held),
    }
