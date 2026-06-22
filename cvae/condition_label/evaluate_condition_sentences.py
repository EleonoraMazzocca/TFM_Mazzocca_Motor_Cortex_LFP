"""Evaluate sentence-encoding strategies for cVAE condition labels.

Encodes 12 condition sentences under five options (A/B/C/D/E) using MiniLM and
scores each on four criteria:
  1. Factor coherence  — within-factor similarity vs cross-factor similarity
  2. Phase gradient    — adjacent phases more similar than distant phases
  3. Held-out proximity — held-out class sits close to its seen neighbours
  4. Linear recoverability — descriptive in-sample identification after PCA

No CVAE, no data, no imports from the rest of the codebase.

Requirements:
    pip install sentence-transformers matplotlib

Usage:
    python evaluate_condition_sentences.py
    python evaluate_condition_sentences.py --out_dir results/sentence_eval
    python evaluate_condition_sentences.py --model all-MiniLM-L12-v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Sentences — 12 conditions, three options
# ---------------------------------------------------------------------------
# Index ordering (0-based throughout):
#   0–3  : prereach + power/precision × left/right
#   4–7  : reach   + power/precision × left/right
#   8–11 : grasp   + power/precision × left/right
# Within each phase block: (power+left, power+right, precision+left, precision+right)

OPTION_A = [
    "preparatory phase before arm transport, power grip, left hand",
    "preparatory phase before arm transport, power grip, right hand",
    "preparatory phase before arm transport, precision grip, left hand",
    "preparatory phase before arm transport, precision grip, right hand",
    "arm transport phase toward object, power grip, left hand",
    "arm transport phase toward object, power grip, right hand",
    "arm transport phase toward object, precision grip, left hand",
    "arm transport phase toward object, precision grip, right hand",
    "object contact and grip closure, power grip, left hand",
    "object contact and grip closure, power grip, right hand",
    "object contact and grip closure, precision grip, left hand",
    "object contact and grip closure, precision grip, right hand",
]

OPTION_B = [
    "arm preparation before movement onset, the first phase in the sequence, whole-hand power grip, left hand, followed by reach and grasp",
    "arm preparation before movement onset, the first phase in the sequence, whole-hand power grip, right hand, followed by reach and grasp",
    "arm preparation before movement onset, the first phase in the sequence, fingertip precision grip, left hand, followed by reach and grasp",
    "arm preparation before movement onset, the first phase in the sequence, fingertip precision grip, right hand, followed by reach and grasp",
    "arm transport toward the object, the second phase following prereach, whole-hand power grip, left hand, preceding final grasp",
    "arm transport toward the object, the second phase following prereach, whole-hand power grip, right hand, preceding final grasp",
    "arm transport toward the object, the second phase following prereach, fingertip precision grip, left hand, preceding final grasp",
    "arm transport toward the object, the second phase following prereach, fingertip precision grip, right hand, preceding final grasp",
    "object contact and grip closure, the final phase following reach, whole-hand power grip, left hand, completing the grasp sequence",
    "object contact and grip closure, the final phase following reach, whole-hand power grip, right hand, completing the grasp sequence",
    "object contact and grip closure, the final phase following reach, fingertip precision grip, left hand, completing the grasp sequence",
    "object contact and grip closure, the final phase following reach, fingertip precision grip, right hand, completing the grasp sequence",
]

OPTION_C = [
    "prereach motor preparation, prehension configuration, power grip, left limb",
    "prereach motor preparation, prehension configuration, power grip, right limb",
    "prereach motor preparation, prehension configuration, precision grip, left limb",
    "prereach motor preparation, prehension configuration, precision grip, right limb",
    "reach limb transport, goal-directed movement, power grip, left limb",
    "reach limb transport, goal-directed movement, power grip, right limb",
    "reach limb transport, goal-directed movement, precision grip, left limb",
    "reach limb transport, goal-directed movement, precision grip, right limb",
    "grasp digit closure, object acquisition, power grip, left limb",
    "grasp digit closure, object acquisition, power grip, right limb",
    "grasp digit closure, object acquisition, precision grip, left limb",
    "grasp digit closure, object acquisition, precision grip, right limb",
]

OPTION_D = [
    "you are ready for a new trial, hold your rest position and prepare to apply a strong full-hand power grip using your left hand",
    "you are ready for a new trial, hold your rest position and prepare to apply a strong full-hand power grip using your right hand",
    "you are ready for a new trial, hold your rest position and prepare to apply a delicate fingertip precision grip using your left hand",
    "you are ready for a new trial, hold your rest position and prepare to apply a delicate fingertip precision grip using your right hand",
    "begin moving your left arm toward the target object in preparation for wrapping all fingers around it in a full-hand power grip",
    "begin moving your right arm toward the target object in preparation for wrapping all fingers around it in a full-hand power grip",
    "begin moving your left arm toward the target object in preparation for pinching it carefully between thumb and index finger in a precision grip",
    "begin moving your right arm toward the target object in preparation for pinching it carefully between thumb and index finger in a precision grip",
    "you have reached the target, now close your entire left hand firmly around it applying maximum contact with all fingers in a power grip",
    "you have reached the target, now close your entire right hand firmly around it applying maximum contact with all fingers in a power grip",
    "you have reached the target, now pinch it precisely between the fingertips of your left hand applying a controlled precision grip",
    "you have reached the target, now pinch it precisely between the fingertips of your right hand applying a controlled precision grip",
]

OPTION_E = [
    "preparatory stillness before movement, left arm stationary, fingers extended ready to curl all five digits into a tight encompassing grip",
    "preparatory stillness before movement, right arm stationary, fingers extended ready to curl all five digits into a tight encompassing grip",
    "preparatory stillness before movement, left arm stationary, thumb and index finger poised for delicate two-digit pinch contact",
    "preparatory stillness before movement, right arm stationary, thumb and index finger poised for delicate two-digit pinch contact",
    "left arm transporting toward object, wrist rotating, hand opening wide to receive object with full palmar contact and all digits",
    "right arm transporting toward object, wrist rotating, hand opening wide to receive object with full palmar contact and all digits",
    "left arm transporting toward object, wrist stabilising, thumb and forefinger forming an aperture sized for precise two-digit contact",
    "right arm transporting toward object, wrist stabilising, thumb and forefinger forming an aperture sized for precise two-digit contact",
    "left hand closes around object, all five fingers flexing simultaneously into maximum palmar grip, completing the reach-to-grasp sequence",
    "right hand closes around object, all five fingers flexing simultaneously into maximum palmar grip, completing the reach-to-grasp sequence",
    "left hand makes final contact, thumb opposing index finger in controlled pinch, completing the reach-to-grasp sequence",
    "right hand makes final contact, thumb opposing index finger in controlled pinch, completing the reach-to-grasp sequence",
]

OPTIONS = {
    "A_template":     OPTION_A,
    "B_narrative":    OPTION_B,
    "C_motor":        OPTION_C,
    "D_instructional": OPTION_D,
    "E_anatomical":   OPTION_E,
}

# ---------------------------------------------------------------------------
# Factor labels (0-based, length 12)
# ---------------------------------------------------------------------------
PHASES = [0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2]   # 0=prereach,1=reach,2=grasp
GRIPS  = [0, 0, 1, 1,  0, 0, 1, 1,  0, 0, 1, 1]   # 0=power,1=precision
HANDS  = [0, 1, 0, 1,  0, 1, 0, 1,  0, 1, 0, 1]   # 0=left,1=right

PHASE_NAMES = ["prereach", "reach", "grasp"]
GRIP_NAMES  = ["power", "precision"]
HAND_NAMES  = ["left", "right"]
CONDITION_NAMES = [
    f"{PHASE_NAMES[PHASES[i]]}+{GRIP_NAMES[GRIPS[i]]}+{HAND_NAMES[HANDS[i]]}"
    for i in range(12)
]

# Held-out: grasp + precision + right → index 11
# Neighbours: grasp+precision+left (10), grasp+power+right (9)
# Distant:   prereach+power+left (0)
HELDOUT_IDX   = 11
NEIGHBOUR_IDX = [10, 9]
DISTANT_IDX   = 0


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def factor_coherence(sim: np.ndarray, labels: list[int]) -> tuple[float, float]:
    """Mean within-factor and cross-factor cosine similarity (upper triangle only)."""
    within, across = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within.append(sim[i, j])
            else:
                across.append(sim[i, j])
    return float(np.mean(within)), float(np.mean(across))


def phase_gradient(embeddings: np.ndarray) -> tuple[float, float, float, bool]:
    """Cosine similarities between mean phase embeddings.

    Returns (pre_reach, reach_grasp, pre_grasp, gradient_holds).
    Gradient holds when adjacent phases are more similar than distant ones:
      pre_reach > pre_grasp AND reach_grasp > pre_grasp.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    phase_embs = np.stack([
        embeddings[0:4].mean(0),
        embeddings[4:8].mean(0),
        embeddings[8:12].mean(0),
    ])
    s = cos_sim(phase_embs)
    pre_reach   = float(s[0, 1])
    reach_grasp = float(s[1, 2])
    pre_grasp   = float(s[0, 2])
    holds = (pre_reach > pre_grasp) and (reach_grasp > pre_grasp)
    return pre_reach, reach_grasp, pre_grasp, holds


def heldout_proximity(sim: np.ndarray) -> tuple[float, float, float]:
    """Similarity between held-out condition and its two nearest seen neighbours vs a distant class."""
    sim_n1 = float(sim[HELDOUT_IDX, NEIGHBOUR_IDX[0]])
    sim_n2 = float(sim[HELDOUT_IDX, NEIGHBOUR_IDX[1]])
    sim_d  = float(sim[HELDOUT_IDX, DISTANT_IDX])
    return sim_n1, sim_n2, sim_d


def pca_dims(embeddings: np.ndarray, thresholds: tuple[float, ...] = (0.95, 0.99)) -> dict[float, int]:
    """Number of PCA components needed to explain each variance threshold.

    With 12 conditions the intrinsic rank is at most 11. This tells you the
    natural condition dimensionality driven by sentence structure, not an
    arbitrary choice like 32.
    """
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    result = {}
    for t in thresholds:
        n = int(np.searchsorted(cumvar, t) + 1)
        result[t] = n
    return result


def linear_recoverability(
    embeddings: np.ndarray,
    dimensions: range = range(2, 8),
) -> dict[str, dict]:
    """Measure descriptive 12-condition linear recoverability after PCA.

    The classifier is fitted and evaluated on the same 12 condition vectors.
    This is deliberately a representation diagnostic, not a cross-validated
    estimate of generalization performance.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    labels = np.arange(len(CONDITION_NAMES), dtype=np.int64)
    max_dim = min(embeddings.shape[0] - 1, embeddings.shape[1])
    records = {}
    for dim in dimensions:
        if dim > max_dim:
            continue
        projected = PCA(n_components=dim, random_state=42).fit_transform(embeddings)
        classifier = LogisticRegression(
            max_iter=10000,
            solver="lbfgs",
            random_state=42,
        )
        classifier.fit(projected, labels)
        predictions = classifier.predict(projected).astype(np.int64)
        errors = [
            {
                "index": int(i),
                "true": CONDITION_NAMES[i],
                "predicted": CONDITION_NAMES[int(predictions[i])],
            }
            for i in range(len(labels))
            if predictions[i] != labels[i]
        ]
        records[str(dim)] = {
            "n_correct": int((predictions == labels).sum()),
            "n_total": int(len(labels)),
            "accuracy": float((predictions == labels).mean()),
            "fully_recoverable": bool(np.all(predictions == labels)),
            "errors": errors,
        }
    return records


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def save_heatmap(sim: np.ndarray, option_name: str, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [
            f"{PHASE_NAMES[PHASES[i]]}\n{GRIP_NAMES[GRIPS[i]]}\n{HAND_NAMES[HANDS[i]]}"
            for i in range(12)
        ]
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim, vmin=0.4, vmax=1.0, cmap="RdYlBu_r")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(12)); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(12)); ax.set_yticklabels(labels, fontsize=6)
        for i in range(12):
            for j in range(12):
                ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                        fontsize=5, color="black" if 0.5 < sim[i,j] < 0.9 else "white")
        # Draw phase block boundaries
        for boundary in [3.5, 7.5]:
            ax.axhline(boundary, color="black", linewidth=1.5)
            ax.axvline(boundary, color="black", linewidth=1.5)
        # Mark held-out cell
        ax.add_patch(plt.Rectangle((HELDOUT_IDX - 0.5, HELDOUT_IDX - 0.5), 1, 1,
                                   fill=False, edgecolor="red", linewidth=2.5))
        ax.set_title(f"Cosine similarity — Option {option_name}", fontsize=11)
        plt.tight_layout()
        path = out_dir / f"sim_heatmap_{option_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path.name}")
    except Exception as e:
        print(f"  WARNING: heatmap failed — {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=str, default=".",
                   help="Directory for .npy matrices and heatmaps (default: current dir).")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                   help="SentenceTransformer model name (default: all-MiniLM-L6-v2).")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}\n"
            "Install with: pip install sentence-transformers scikit-learn"
        )

    print(f"Loading {args.model} ...")
    lm = SentenceTransformer(args.model)

    results = {}
    recoverability_results = {}
    for opt_name, sentences in OPTIONS.items():
        print(f"\nEncoding option {opt_name} ...")
        emb = lm.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        sim = cosine_similarity(emb)

        np.save(out_dir / f"sim_{opt_name}.npy", sim)
        np.save(out_dir / f"emb_{opt_name}.npy", emb)

        # Save PCA-projected condition table for Option D (the selected option).
        if opt_name == "D_instructional":
            from sklearn.decomposition import PCA as _PCA
            _pca = _PCA(n_components=5, random_state=42)
            _projected = _pca.fit_transform(emb).astype(np.float32)  # (12, 5)
            _key_order = np.array(
                [[PHASES[i], GRIPS[i], HANDS[i]] for i in range(12)], dtype=np.int64
            )  # (12, 3) — (phase_id, grip_id, hand_id)
            np.save(out_dir / "condition_vectors_D_pca5.npy", _projected)
            np.save(out_dir / "condition_keys_D_pca5.npy",    _key_order)
            print(f"  Saved condition_vectors_D_pca5.npy  shape={_projected.shape}")
            print(f"  Saved condition_keys_D_pca5.npy     shape={_key_order.shape}")

        ph_in,  ph_out  = factor_coherence(sim, PHASES)
        gr_in,  gr_out  = factor_coherence(sim, GRIPS)
        ha_in,  ha_out  = factor_coherence(sim, HANDS)
        pre_reach, reach_grasp, pre_grasp, gradient = phase_gradient(emb)
        sim_n1, sim_n2, sim_d = heldout_proximity(sim)

        pca_d = pca_dims(emb)
        recoverability_results[opt_name] = linear_recoverability(emb)

        results[opt_name] = {
            "phase_in": ph_in,  "phase_out": ph_out,
            "grip_in":  gr_in,  "grip_out":  gr_out,
            "hand_in":  ha_in,  "hand_out":  ha_out,
            "pre_reach": pre_reach, "reach_grasp": reach_grasp, "pre_grasp": pre_grasp,
            "phase_gradient": gradient,
            "held_n1": sim_n1, "held_n2": sim_n2, "held_distant": sim_d,
            "pca_dims_95": pca_d[0.95],
            "pca_dims_99": pca_d[0.99],
        }
        save_heatmap(sim, opt_name, out_dir)

    # --- Compact comparison table ---
    W = 128
    print("\n" + "=" * W)
    print(f"{'option':<17} "
          f"{'ph_gap':>7} {'gr_gap':>7} {'ha_gap':>7}  "
          f"{'pre↔rch':>8} {'rch↔grsp':>9} {'pre↔grsp':>9}  "
          f"{'grad':>5}  "
          f"{'hld↔n1':>8} {'hld↔n2':>8} {'hld↔dst':>8}  "
          f"{'pca@95':>7} {'pca@99':>7}")
    print("-" * W)
    for opt_name, r in results.items():
        grad_str = "YES" if r["phase_gradient"] else "NO "
        print(f"{opt_name:<17} "
              f"{r['phase_in']-r['phase_out']:>+7.3f} "
              f"{r['grip_in']-r['grip_out']:>+7.3f} "
              f"{r['hand_in']-r['hand_out']:>+7.3f}  "
              f"{r['pre_reach']:>8.3f} {r['reach_grasp']:>9.3f} {r['pre_grasp']:>9.3f}  "
              f"{grad_str:>5}  "
              f"{r['held_n1']:>8.3f} {r['held_n2']:>8.3f} {r['held_distant']:>8.3f}  "
              f"{r['pca_dims_95']:>7d} {r['pca_dims_99']:>7d}")
    print("=" * W)

    print("\n12-condition linear recoverability (fit and evaluated on the same 12 vectors):")
    for opt_name, records in recoverability_results.items():
        intrinsic_dim = results[opt_name]["pca_dims_99"]
        print(f"\n{opt_name}  (pca@99 = {intrinsic_dim} dims)")
        for dim, record in records.items():
            marker = " ← pca@99" if int(dim) == intrinsic_dim else ""
            status = "SEPARABLE" if record["fully_recoverable"] else "not fully separable"
            print(
                f"  {dim} dims: training accuracy={record['accuracy']:.4f}  {status}{marker}"
            )
        intrinsic_record = records.get(str(intrinsic_dim))
        if intrinsic_record and intrinsic_record["errors"]:
            print("  Confusions at pca@99:")
            for error in intrinsic_record["errors"]:
                print(f"    true: {error['true']:<36} predicted: {error['predicted']}")

    # --- Interpretation ---
    print("\nColumn guide:")
    print("  ph/gr/ha gap : within-factor − cross-factor similarity (positive = good, >0.15 is meaningful)")
    print("  pre↔rch etc  : cosine similarity between mean phase embeddings")
    print("  grad         : YES if adjacent phases are more similar than distant phases")
    print("  hld↔n1/n2    : similarity to nearest seen neighbours (want >0.85)")
    print("  hld↔dst      : similarity to most distant condition (want <0.65)")
    print("  pca@95/99    : components to explain 95%/99% variance → natural condition dim")
    print()
    print("Recommendation:")
    best = max(results, key=lambda k: (
        results[k]["phase_in"] - results[k]["phase_out"]
        + results[k]["grip_in"] - results[k]["grip_out"]
        + results[k]["hand_in"] - results[k]["hand_out"]
        + int(results[k]["phase_gradient"]) * 0.1
        + results[k]["held_n1"] + results[k]["held_n2"]
        - results[k]["held_distant"]
    ))
    print(f"  Highest composite score: {best}")
    print(f"  Phase gradient: " + "  ".join(
        f"{k}={'YES' if v['phase_gradient'] else 'NO'}" for k, v in results.items()
    ))
    print(f"  Suggested condition dim (pca@99): " + "  ".join(
        f"{k}={v['pca_dims_99']}" for k, v in results.items()
    ))

    # --- Save summary JSON ---
    import json
    (out_dir / "sentence_eval_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (out_dir / "condition_linear_recoverability.json").write_text(
        json.dumps(recoverability_results, indent=2), encoding="utf-8"
    )
    print(
        f"Saved linear-recoverability records to {out_dir / 'condition_linear_recoverability.json'}"
    )
    print(f"\nSaved matrices, heatmaps, and summary to {out_dir}")


if __name__ == "__main__":
    main()
