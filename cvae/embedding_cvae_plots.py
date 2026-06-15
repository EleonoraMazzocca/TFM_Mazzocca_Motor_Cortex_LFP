"""Plotting helpers for embedding-space cVAE experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from transformer_encoder.joint_embedding_data import GRIP_TO_ID, HAND_TO_ID, PHASE_NAMES

ID_TO_GRIP = {v: k for k, v in GRIP_TO_ID.items()}
ID_TO_HAND = {v: k for k, v in HAND_TO_ID.items()}


def plot_generation_diagnostics(plot_data: dict, out_dir: Path, args) -> None:
    """Save per-run generation plots from precomputed evaluation artifacts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping generation plots")
        return

    x_train = plot_data["x_train"]
    x_real = plot_data["x_real"]
    x_gen = plot_data["x_gen"]
    train_payload = plot_data["train_payload"]
    target_phase = plot_data["target_phase"]
    target_grip = plot_data["target_grip"]
    target_hand = plot_data["target_hand"]
    target_combo = plot_data["target_combo"]
    centroid_keys = plot_data["centroid_keys"]
    centroid_mat = plot_data["centroid_mat"]
    gen_centroid = plot_data["gen_centroid"]
    target_centroid = plot_data["target_centroid"]
    nearest_target_rate = plot_data["nearest_target_rate"]
    centroid_dist = plot_data["centroid_dist"]
    target_to_other_min = plot_data["target_to_other_min"]
    target_to_other_mean = plot_data["target_to_other_mean"]
    relative_centroid_distance_mean = plot_data["relative_centroid_distance_mean"]
    relative_centroid_distance_min = plot_data["relative_centroid_distance_min"]
    centroid_distance_table = plot_data["centroid_distance_table"]
    head_accuracy = plot_data["head_accuracy"]
    preds_gen = plot_data["preds_gen"]
    preds_real = plot_data["preds_real"]
    head_targets = plot_data["head_targets"]
    head_class_names = plot_data["head_class_names"]

    try:
        pca = PCA(n_components=3, random_state=args.seed)
        pca.fit(x_train)
        tr_pc = pca.transform(x_train)
        real_pc = pca.transform(x_real)
        gen_pc = pca.transform(x_gen)
        centroid_pc = pca.transform(centroid_mat)
        gen_centroid_pc = pca.transform(gen_centroid[None, :])
        target_centroid_pc = pca.transform(target_centroid[None, :])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, xi, yi, ylabel in [(axes[0], 0, 1, "PC2"), (axes[1], 0, 2, "PC3")]:
            ax.scatter(tr_pc[:, xi], tr_pc[:, yi], c="grey", alpha=0.12, s=8, label="seen train")
            ax.scatter(real_pc[:, xi], real_pc[:, yi], c="#4C72B0", alpha=0.75, s=20, label="real held-out")
            ax.scatter(gen_pc[:, xi], gen_pc[:, yi], c="#DD8452", alpha=0.75, s=20, label="generated")
            ax.set_xlabel("PC1")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        fig.suptitle("Embedding cVAE generation")
        plt.tight_layout()
        plt.savefig(out_dir / "embedding_generation_pca.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, xi, yi, ylabel in [(axes[0], 0, 1, "PC2"), (axes[1], 0, 2, "PC3")]:
            ax.scatter(tr_pc[:, xi], tr_pc[:, yi], c="grey", alpha=0.06, s=6, label="seen train")
            for row, combo in enumerate(centroid_keys):
                color = "#4C72B0" if combo == target_combo else "#222222"
                marker = "*" if combo == target_combo else "o"
                size = 170 if combo == target_combo else 55
                ax.scatter(
                    centroid_pc[row, xi], centroid_pc[row, yi],
                    c=color, marker=marker, s=size, edgecolors="white", linewidths=0.7,
                    label="target centroid" if combo == target_combo and xi == 0 and yi == 1 else None,
                )
                if combo != target_combo and xi == 0 and yi == 1:
                    ax.text(centroid_pc[row, xi], centroid_pc[row, yi], str(int(combo)), fontsize=6, alpha=0.8)
            ax.scatter(
                gen_centroid_pc[0, xi], gen_centroid_pc[0, yi],
                c="#DD8452", marker="X", s=150, edgecolors="white", linewidths=0.7,
                label="generated centroid",
            )
            ax.plot(
                [target_centroid_pc[0, xi], gen_centroid_pc[0, xi]],
                [target_centroid_pc[0, yi], gen_centroid_pc[0, yi]],
                color="#DD8452", linewidth=1.5, alpha=0.8,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        fig.suptitle("Embedding centroids in PCA space")
        plt.tight_layout()
        plt.savefig(out_dir / "embedding_centroids_pca.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(["target centroid"], [nearest_target_rate], color=["#937860"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Generated samples nearest to held-out centroid")
        ax.set_title("Embedding cVAE centroid assignment")
        ax.text(bars[0].get_x() + bars[0].get_width() / 2, nearest_target_rate + 0.03, f"{nearest_target_rate:.2f}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / "embedding_centroid_assignment.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["gen→target", "target→nearest other", "target→mean other"]
        vals = [centroid_dist, target_to_other_min, target_to_other_mean]
        colors = ["#DD8452", "#7F7F7F", "#4C72B0"]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_ylabel("Euclidean distance in normalized embedding space")
        ax.set_title("Centroid distance context")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.03, f"{val:.2f}", ha="center", fontsize=9)
        ax.text(
            0.02, 0.95,
            f"relative to mean other = {relative_centroid_distance_mean:.2f}\n"
            f"relative to nearest other = {relative_centroid_distance_min:.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        plt.tight_layout()
        plt.savefig(out_dir / "embedding_centroid_distance_context.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 9))
        target_label = f"{args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}"
        rows = centroid_distance_table + [{
            "combo": f"generated {target_label}",
            "distance_to_target": centroid_dist,
            "closer_than_generated": False,
            "is_generated": True,
        }]
        rows = sorted(rows, key=lambda row: row["distance_to_target"])
        labels = [row["combo"] for row in rows]
        vals = [row["distance_to_target"] for row in rows]
        colors = [
            "#DD8452" if row.get("is_generated") else "#C44E52" if row["closer_than_generated"] else "#7F7F7F"
            for row in rows
        ]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(centroid_dist, color="#DD8452", linewidth=2, label="generated→target")
        ax.set_ylim(0, max(vals) * 1.08)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Distance to real held-out target centroid")
        ax.set_title("Distance to held-out target centroid")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "embedding_target_centroid_distances.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        if head_accuracy is not None:
            factors = ["phase", "grip", "hand"]
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            ax = axes[0]
            x_pos = np.arange(len(factors))
            width = 0.32
            bars_gen = ax.bar(x_pos - width / 2, [head_accuracy["generated"][f] for f in factors], width, label="generated", color="#DD8452")
            bars_real = ax.bar(x_pos + width / 2, [head_accuracy["real_heldout"][f] for f in factors], width, label="real held-out", color="#4C72B0")
            for bar in list(bars_gen) + list(bars_real):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha="center", fontsize=9)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(factors)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Accuracy")
            ax.set_title("Head accuracy: generated vs real held-out")
            ax.legend()
            ax = axes[1]
            ax.axis("off")
            table_data = []
            for f in factors:
                n_cls = len(head_class_names[f])
                gen_top = int(np.bincount(preds_gen[f], minlength=n_cls).argmax())
                real_top = int(np.bincount(preds_real[f], minlength=n_cls).argmax())
                table_data.append([
                    f,
                    head_class_names[f][head_targets[f]],
                    f"{head_accuracy['generated'][f]:.2f}",
                    f"{head_accuracy['real_heldout'][f]:.2f}",
                    head_class_names[f][gen_top],
                    head_class_names[f][real_top],
                ])
            tbl = ax.table(cellText=table_data, colLabels=["factor", "true class", "gen acc", "real acc", "gen top pred", "real top pred"], loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.8)
            ax.set_title("Prediction summary", pad=20)
            fig.suptitle(f"Transformer head accuracy — {args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand}")
            plt.tight_layout()
            plt.savefig(out_dir / "embedding_head_accuracy.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_dir / 'embedding_head_accuracy.png'}")

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for ax, factor in zip(axes, factors):
                cls_names = head_class_names[factor]
                n_cls = len(cls_names)
                x_cls = np.arange(n_cls)
                w = 0.35
                dist_gen = np.array([ (preds_gen[factor] == c).mean() for c in range(n_cls) ])
                dist_real = np.array([ (preds_real[factor] == c).mean() for c in range(n_cls) ])
                ax.bar(x_cls - w / 2, dist_gen, w, label="generated", color="#DD8452", alpha=0.85)
                ax.bar(x_cls + w / 2, dist_real, w, label="real held-out", color="#4C72B0", alpha=0.85)
                ax.axvline(head_targets[factor], color="black", linewidth=1.5, linestyle="--", label=f"true: {cls_names[head_targets[factor]]}")
                ax.set_xticks(x_cls)
                ax.set_xticklabels(cls_names)
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Fraction of samples")
                ax.set_title(factor)
                ax.legend(fontsize=8)
            fig.suptitle(
                f"Predicted class distribution — generated vs real held-out\n"
                f"({args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand})"
            )
            plt.tight_layout()
            plt.savefig(out_dir / "embedding_head_class_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_dir / 'embedding_head_class_distribution.png'}")

        grip_names = ["power", "precision"]
        hand_names = ["left", "right"]
        gh_cols = [(0, 0), (0, 1), (1, 0), (1, 1)]
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        for ph in range(len(PHASE_NAMES)):
            for col, (gr, ha) in enumerate(gh_cols):
                ax = axes[ph, col]
                is_heldout = (ph == target_phase and gr == target_grip and ha == target_hand)
                cond_name = f"{PHASE_NAMES[ph]}+{grip_names[gr]}+{hand_names[ha]}"
                if is_heldout:
                    ax.scatter(real_pc[:, 0], real_pc[:, 1], c="#4C72B0", alpha=0.55, s=12, linewidths=0, label="real held-out")
                    ax.scatter(gen_pc[:, 0], gen_pc[:, 1], c="#DD8452", alpha=0.55, s=12, linewidths=0, label="generated")
                    ax.set_facecolor("#FFF0F0")
                    ax.set_title(f"{cond_name}\n[TARGET — held-out]", fontsize=7, color="#C44E52")
                    ax.legend(fontsize=6, markerscale=2, loc="best")
                else:
                    mask = (
                        (train_payload["y_phase"] == ph)
                        & (train_payload["y_grip"] == gr)
                        & (train_payload["y_hand"] == ha)
                    )
                    if mask.any():
                        cond_pc = pca.transform(x_train[mask])
                        ax.scatter(cond_pc[:, 0], cond_pc[:, 1], c="#4C72B0", alpha=0.4, s=10, linewidths=0, label="seen condition")
                    ax.scatter(gen_pc[:, 0], gen_pc[:, 1], c="#DD8452", alpha=0.25, s=8, linewidths=0, label="generated")
                    ax.set_title(cond_name, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(
            f"Generated ({args.heldout_phase}+{args.heldout_grip}+{args.heldout_hand})"
            f" overlaid on each seen condition\n"
            f"Orange = generated  |  Blue = seen condition  |  Red cell = true target",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(out_dir / "embedding_generated_vs_conditions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / 'embedding_generated_vs_conditions.png'}")

    except Exception as e:
        print(f"  WARNING: generation plots failed: {e}")


def _load_mmd_seen(data: dict) -> dict:
    return json.loads(data["mmd_seen"].item())


def generate_comparison_plots(baseline_dirs: list[str], aug_dirs: list[str], out_dir: Path) -> None:
    """Load diagnostics from completed run dirs and generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping comparison plots")
        return

    def _load_dir(d: str) -> tuple[dict, dict, dict, Path]:
        dp = Path(d)
        diag = dict(np.load(dp / "collapse_diagnostics.npz", allow_pickle=False))
        kl = dict(np.load(dp / "kl_history.npz", allow_pickle=False))
        try:
            ckpt = torch.load(dp / "checkpoint.pt", map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(dp / "checkpoint.pt", map_location="cpu")
        return diag, kl, ckpt, dp

    def _aug_group_label(ckpt: dict) -> str:
        a = ckpt.get("args", {})
        denoising = bool(a.get("denoising_aug", False))
        cond_drop = bool(a.get("cond_dropout", False))
        scale = a.get("amplitude_scale_range", [0.85, 1.15])
        parts = []
        if denoising:
            parts.append(f"denoising: n_drop={a.get('aug_n_dropout_dims', 2)}, noise={a.get('noise_scale', 0.1)}, scale=({scale[0]},{scale[1]})")
        if cond_drop:
            p_single = a.get("p_cond_single", 0.15)
            p_double = a.get("p_cond_double", 0.04)
            p_all = a.get("p_cond_all", 0.03)
            p_full = 1 - 3 * p_single - 3 * p_double - p_all
            parts.append(f"cond_drop: p1={p_single}, p2={p_double}, p_all={p_all}, p_full={p_full:.2f}")
        fb = float(a.get("free_bits", 0.0))
        if fb > 0.0:
            parts.append(f"free_bits={fb}")
        beta = float(a.get("beta_max", 1.0))
        if beta != 1.0:
            parts.append(f"beta={beta}")
        if a.get("mmd_loss", False):
            parts.append(f"mmd=True, lam={a.get('lambda_mmd', 10.0)}")
        return " + ".join(parts) if parts else "no augmentation"

    all_baseline = [_load_dir(d) for d in baseline_dirs]
    all_aug = [_load_dir(d) for d in aug_dirs]
    all_runs = all_baseline + all_aug
    aug_groups: dict[str, list] = {}
    for item in all_aug:
        key = _aug_group_label(item[2])
        aug_groups.setdefault(key, []).append(item)
    aug_group_keys = sorted(aug_groups.keys())
    all_groups = [("baseline", all_baseline)] + [(gkey, aug_groups[gkey]) for gkey in aug_group_keys]
    n_groups = len(all_groups)

    ref_diag = all_runs[0][0]
    ref_split_seed = int(ref_diag["split_seed"])
    ref_latent_dim = int(ref_diag["latent_dim"])
    ref_gen_seed = int(ref_diag["generation_seed"])
    ref_heldout = ref_diag["heldout_combo"].tolist()
    ref_val_indices = ref_diag["val_indices"]
    ref_n_epochs = len(all_runs[0][1]["val_kl_mean_per_dim"])
    ref_bw = _load_mmd_seen(ref_diag)
    ref_ckpt_args = all_runs[0][2].get("args", {})

    errors = []
    for diag, kl, ckpt, run_dir in all_runs[1:]:
        label = str(run_dir)
        a = ckpt.get("args", {})
        if int(diag["split_seed"]) != ref_split_seed:
            errors.append(f"{label}: split_seed mismatch")
        if int(diag["latent_dim"]) != ref_latent_dim:
            errors.append(f"{label}: latent_dim mismatch")
        if int(diag["generation_seed"]) != ref_gen_seed:
            errors.append(f"{label}: generation_seed mismatch")
        if diag["heldout_combo"].tolist() != ref_heldout:
            errors.append(f"{label}: heldout_combo mismatch")
        if not np.array_equal(diag["val_indices"], ref_val_indices):
            errors.append(f"{label}: val_indices mismatch — validation sets differ")
        if len(kl["val_kl_mean_per_dim"]) != ref_n_epochs:
            errors.append(f"{label}: epoch count mismatch ({len(kl['val_kl_mean_per_dim'])} vs {ref_n_epochs})")
        run_bw = _load_mmd_seen(diag)
        if set(run_bw) != set(ref_bw):
            errors.append(f"{label}: seen-class MMD combination keys differ from reference")
        else:
            for combo in ref_bw:
                if not np.isclose(run_bw[combo]["bandwidth"], ref_bw[combo]["bandwidth"], atol=1e-6, rtol=1e-6):
                    errors.append(f"{label}: MMD bandwidth mismatch for {combo}")
        if a.get("joint_checkpoint") != ref_ckpt_args.get("joint_checkpoint"):
            errors.append(f"{label}: joint_checkpoint mismatch")
        if a.get("hidden_dims") != ref_ckpt_args.get("hidden_dims"):
            errors.append(f"{label}: hidden_dims mismatch")
        if not np.isclose(float(diag["mmd_heldout_bandwidth"]), float(ref_diag["mmd_heldout_bandwidth"]), atol=1e-6, rtol=1e-6):
            errors.append(f"{label}: held-out MMD bandwidth mismatch")
    if errors:
        print("  Compatibility check FAILED — cannot generate comparison plots:")
        for e in errors:
            print(f"    {e}")
        return

    epochs = np.arange(1, ref_n_epochs + 1)
    cmap_tab10 = plt.get_cmap("tab10")
    cmap_tab20 = plt.get_cmap("tab20")
    group_colors: dict[str, object] = {"baseline": "#4C72B0"}
    for i, gkey in enumerate(aug_group_keys):
        group_colors[gkey] = cmap_tab10(i % 10)
    w = min(0.7 / max(n_groups, 1), 0.35)
    offsets = np.linspace(-(n_groups - 1) * w / 2, (n_groups - 1) * w / 2, n_groups)

    diag_keys = [
        ("std_mu", "std(mu) per latent dim\n[PRIMARY collapse detector]"),
        ("mean_sigma", "mean(sigma) per latent dim"),
        ("second_moment", "E[mu²+exp(log_var)] per latent dim\n[secondary — not reliable alone]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, title) in zip(axes, diag_keys):
        x_dim = np.arange(ref_latent_dim)
        for glabel, group_runs in all_groups:
            curves = [np.sort(diag[key])[::-1] for diag, _, _, _ in group_runs]
            arr = np.stack(curves)
            mean_ = arr.mean(axis=0)
            std_ = arr.std(axis=0)
            ax.plot(x_dim, mean_, color=group_colors[glabel], label=glabel, linewidth=1.5)
            ax.fill_between(x_dim, mean_ - std_, mean_ + std_, color=group_colors[glabel], alpha=0.25)
        ax.set_xlabel("Latent dimension (sorted per seed)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle("Collapse diagnostics: baseline vs augmented", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "collapse_diagnostics_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved collapse_diagnostics_panel.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, kl_key, title in [(axes[0], "val_kl_mean_per_dim", "Val KL mean per dim"), (axes[1], "val_kl_sum_dims", "Val KL summed across dims")]:
        for glabel, group_runs in all_groups:
            curves = [kl[kl_key] for _, kl, _, _ in group_runs]
            arr = np.stack(curves)
            mean_ = arr.mean(axis=0)
            std_ = arr.std(axis=0)
            ax.plot(epochs, mean_, color=group_colors[glabel], label=glabel, linewidth=1.5)
            ax.fill_between(epochs, mean_ - std_, mean_ + std_, color=group_colors[glabel], alpha=0.25)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.suptitle("KL trajectory over epochs", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "kl_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved kl_trajectory.png")

    ref_combos = sorted(_load_mmd_seen(ref_diag).keys())
    x_pos = np.arange(len(ref_combos))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(ref_combos) * 1.2), 10))
    ax = axes[0]
    for gi, (glabel, group_runs) in enumerate(all_groups):
        gm = np.array([np.mean([_load_mmd_seen(d)[c]["mmd"] for d, _, _, _ in group_runs if c in _load_mmd_seen(d)]) for c in ref_combos])
        gs_ = np.array([np.std([_load_mmd_seen(d)[c]["mmd"] for d, _, _, _ in group_runs if c in _load_mmd_seen(d)]) for c in ref_combos])
        ax.bar(x_pos + offsets[gi], gm, w, yerr=gs_, label=glabel, color=group_colors[glabel], capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ref_combos, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MMD")
    ax.set_title("Seen-class MMD (mean ± std across seeds)")
    ax.legend()

    ax = axes[1]
    for gi, (glabel, group_runs) in enumerate(all_groups):
        held_vals = [float(d["mmd_heldout"]) for d, _, _, _ in group_runs]
        hm = float(np.mean(held_vals))
        hs = float(np.std(held_vals))
        ax.bar([offsets[gi]], [hm], w, yerr=[[hs]], label=glabel, color=group_colors[glabel], capsize=6)
    held_label = "+".join(str(v) for v in ref_diag["heldout_combo"].tolist())
    ax.set_xticks([0])
    ax.set_xticklabels([held_label])
    ax.set_ylabel("MMD")
    ax.set_title("Held-out class MMD (mean ± std across seeds)")
    ax.legend()
    plt.suptitle("MMD comparison: seen vs held-out (never merged)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "mmd_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved mmd_comparison.png")

    ref_mu = all_baseline[0][0]["mu_all"]
    ref_combos_v = all_baseline[0][0]["val_combo_labels"]
    pca_ref = PCA(n_components=2, random_state=0)
    pca_ref.fit(ref_mu)
    unique_c = np.unique(ref_combos_v)

    def _plot_aligned(ax, runs_data, title):
        all_centroids: dict[int, list] = {int(c): [] for c in unique_c}
        for ri, (diag, _, _, _) in enumerate(runs_data):
            run_mu = diag["mu_all"]
            R, _ = orthogonal_procrustes(run_mu, ref_mu)
            aligned = run_mu @ R
            pc = pca_ref.transform(aligned)
            combo_v = diag["val_combo_labels"]
            for ci, combo in enumerate(unique_c):
                mask = combo_v == combo
                ax.scatter(pc[mask, 0], pc[mask, 1], c=[cmap_tab20(ci % 20)], alpha=0.3, s=6, label=str(combo) if ri == 0 else None)
                if mask.any():
                    all_centroids[int(combo)].append((float(pc[mask, 0].mean()), float(pc[mask, 1].mean())))
        for ci, combo in enumerate(unique_c):
            pts = all_centroids[int(combo)]
            if pts:
                cx = float(np.mean([p[0] for p in pts]))
                cy = float(np.mean([p[1] for p in pts]))
                ax.scatter(cx, cy, c=[cmap_tab20(ci % 20)], s=80, marker="*", edgecolors="black", linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=6, ncol=2, title="combo", title_fontsize=6)

    fig, axes = plt.subplots(1, n_groups, figsize=(8 * n_groups, 6), squeeze=False)
    for ax, (glabel, group_runs) in zip(axes[0], all_groups):
        _plot_aligned(ax, group_runs, glabel)
    plt.suptitle("Latent space 2D PCA with Procrustes alignment", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "latent_pca_procrustes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved latent_pca_procrustes.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for diag, kl, ckpt, run_dir in all_baseline:
        best_recon = float(ckpt.get("best_val_recon", float("nan")))
        best_kl = float(ckpt.get("best_val_kl_sum", float("nan")))
        ax.scatter(best_recon, best_kl, c=[group_colors["baseline"]], s=80, marker="o", edgecolors="black", linewidths=0.6, label="baseline (n_drop=0)")
    for gkey in aug_group_keys:
        for diag, kl, ckpt, run_dir in aug_groups[gkey]:
            best_recon = float(ckpt.get("best_val_recon", float("nan")))
            best_kl = float(ckpt.get("best_val_kl_sum", float("nan")))
            ax.scatter(best_recon, best_kl, c=[group_colors[gkey]], s=80, marker="^", edgecolors="black", linewidths=0.6, label=gkey)
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.set_xlabel("Best val reconstruction loss")
    ax.set_ylabel("Best val KL (summed across dims)")
    ax.set_title("Reconstruction vs KL tradeoff\n(o=baseline, ^=augmented; colour=aug config)")
    plt.tight_layout()
    plt.savefig(out_dir / "recon_vs_kl_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved recon_vs_kl_scatter.png")
    print(f"\nAll comparison plots saved to {out_dir}")
