"""Score external condition-sentence candidates for cVAE conditioning.

Input JSON format:
[
  {
    "candidate_id": "llm_001",
    "sentences": ["...", "...", 12 total strings]
  }
]

The row order must be:
  0 prereach power left      1 prereach power right
  2 prereach precision left  3 prereach precision right
  4 reach power left         5 reach power right
  6 reach precision left     7 reach precision right
  8 grasp power left         9 grasp power right
 10 grasp precision left    11 grasp precision right

Usage:
  python score_candidates.py conditions_parallel.json
  python score_candidates.py conditions_parallel.json --cross
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PHASE = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
GRIP = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
HAND = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
KEY_ORDER = np.stack([PHASE, GRIP, HAND], axis=1)
HELDOUT_IDX = 11

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def load_candidates(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Candidate file must contain a JSON list")
    out = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate {i} must be an object")
        cid = str(item.get("candidate_id", f"candidate_{i:03d}"))
        sentences = item.get("sentences")
        if not isinstance(sentences, list) or len(sentences) != 12:
            raise ValueError(f"{cid}: expected exactly 12 sentences")
        if not all(isinstance(s, str) and s.strip() for s in sentences):
            raise ValueError(f"{cid}: all sentences must be non-empty strings")
        out.append({"candidate_id": cid, "sentences": sentences})
    return out


def factor_score(x: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    clf = LogisticRegression(max_iter=10000, solver="lbfgs")

    loo_preds = []
    for train_idx, test_idx in LeaveOneOut().split(x):
        clf.fit(x[train_idx], y[train_idx])
        loo_preds.append(int(clf.predict(x[test_idx])[0]))
    loo_preds = np.asarray(loo_preds, dtype=np.int64)
    loo_correct = int((loo_preds == y).sum())

    clf.fit(x, y)
    train_correct = int((clf.predict(x) == y).sum())

    mask = np.arange(len(y)) != HELDOUT_IDX
    clf.fit(x[mask], y[mask])
    heldout_pred = int(clf.predict(x[[HELDOUT_IDX]])[0])
    proba = clf.predict_proba(x[[HELDOUT_IDX]])[0]
    class_to_col = {int(cls): i for i, cls in enumerate(clf.classes_)}
    true_label = int(y[HELDOUT_IDX])
    true_prob = float(proba[class_to_col[true_label]])
    max_prob = float(np.max(proba))

    return {
        "train_correct": train_correct,
        "loo_correct": loo_correct,
        "train_acc": train_correct / 12.0,
        "loo_acc": loo_correct / 12.0,
        "heldout_pred": heldout_pred,
        "heldout_true": true_label,
        "heldout_correct": bool(heldout_pred == true_label),
        "heldout_true_prob": true_prob,
        "heldout_max_prob": max_prob,
    }


def score_embedding(emb: np.ndarray, dim: int) -> tuple[np.ndarray, dict]:
    from sklearn.decomposition import PCA

    x = PCA(n_components=dim, random_state=42).fit_transform(emb).astype(np.float32)
    scores = {
        "phase": factor_score(x, PHASE),
        "grip": factor_score(x, GRIP),
        "hand": factor_score(x, HAND),
    }
    return x, scores


def dim_pass_basic(scores: dict) -> bool:
    return (
        scores["phase"]["loo_correct"] >= 6
        and scores["grip"]["loo_correct"] >= 5
        and scores["hand"]["loo_correct"] >= 3
        and scores["phase"]["heldout_pred"] == 2
    )


def dim_pass_strong(scores: dict) -> bool:
    return (
        scores["phase"]["loo_correct"] >= 8
        and scores["grip"]["loo_correct"] >= 8
        and 3 <= scores["hand"]["loo_correct"] <= 8
        and scores["phase"]["heldout_pred"] == 2
        and scores["grip"]["heldout_true_prob"] >= 0.45
        and scores["hand"]["heldout_true_prob"] >= 0.45
    )


def row_from_scores(candidate_id: str, model_name: str, dim: int, scores: dict) -> dict:
    row = {
        "candidate_id": candidate_id,
        "model": model_name,
        "dim": dim,
        "basic_dim_pass": dim_pass_basic(scores),
        "strong_dim_pass": dim_pass_strong(scores),
    }
    for factor in ("phase", "grip", "hand"):
        for key, value in scores[factor].items():
            row[f"{factor}_{key}"] = value
    return row


def rank_key(row: dict) -> tuple:
    return (
        int(row["strong_stable"]),
        int(row["basic_stable"]),
        int(row.get("cross_stable", False)),
        row["strong_pass_dims_total"],
        row["basic_pass_dims_total"],
        row["best_phase_loo_correct"] + row["best_grip_loo_correct"] + row["best_hand_loo_correct"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=str)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--cross", action="store_true", help="Also score all-mpnet-base-v2.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args(argv)

    candidates_path = Path(args.candidates_json)
    candidates = load_candidates(candidates_path)
    out_dir = Path(args.out_dir) if args.out_dir else candidates_path.parent / "candidate_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(f"Missing sentence-transformers: {exc}") from exc

    model_names = ["minilm", "mpnet"] if args.cross else ["minilm"]
    all_rows = []
    summary_by_candidate: dict[str, dict] = {
        c["candidate_id"]: {
            "candidate_id": c["candidate_id"],
            "basic_pass_dims_total": 0,
            "strong_pass_dims_total": 0,
            "best_model": None,
            "best_dim": None,
            "best_phase_loo_correct": -1,
            "best_grip_loo_correct": -1,
            "best_hand_loo_correct": -1,
            "basic_stable": False,
            "strong_stable": False,
            "cross_stable": False,
        }
        for c in candidates
    }
    embeddings_by_model = {}

    for short_name in model_names:
        print(f"Loading {MODELS[short_name]} ...")
        model = SentenceTransformer(MODELS[short_name])
        model_rows = []
        for candidate in candidates:
            cid = candidate["candidate_id"]
            emb = model.encode(candidate["sentences"], normalize_embeddings=True, show_progress_bar=False)
            embeddings_by_model[(short_name, cid)] = emb.astype(np.float32)
            for dim in args.dims:
                _, scores = score_embedding(emb, dim)
                row = row_from_scores(cid, short_name, dim, scores)
                model_rows.append(row)
                all_rows.append(row)

        for candidate in candidates:
            cid = candidate["candidate_id"]
            rows = [r for r in model_rows if r["candidate_id"] == cid]
            basic_n = sum(bool(r["basic_dim_pass"]) for r in rows)
            strong_n = sum(bool(r["strong_dim_pass"]) for r in rows)
            s = summary_by_candidate[cid]
            s[f"{short_name}_basic_pass_dims"] = basic_n
            s[f"{short_name}_strong_pass_dims"] = strong_n
            s[f"{short_name}_basic_stable"] = basic_n >= 3
            s[f"{short_name}_strong_stable"] = strong_n >= 3
            s["basic_pass_dims_total"] += basic_n
            s["strong_pass_dims_total"] += strong_n

            best = max(
                rows,
                key=lambda r: (
                    int(r["strong_dim_pass"]),
                    int(r["basic_dim_pass"]),
                    r["phase_loo_correct"] + r["grip_loo_correct"] + r["hand_loo_correct"],
                    r["phase_loo_correct"],
                    r["grip_loo_correct"],
                    r["hand_loo_correct"],
                ),
            )
            if (
                best["phase_loo_correct"] + best["grip_loo_correct"] + best["hand_loo_correct"]
                > s["best_phase_loo_correct"] + s["best_grip_loo_correct"] + s["best_hand_loo_correct"]
            ):
                s["best_model"] = short_name
                s["best_dim"] = best["dim"]
                s["best_phase_loo_correct"] = best["phase_loo_correct"]
                s["best_grip_loo_correct"] = best["grip_loo_correct"]
                s["best_hand_loo_correct"] = best["hand_loo_correct"]
                for factor in ("phase", "grip", "hand"):
                    s[f"best_{factor}_heldout_pred"] = best[f"{factor}_heldout_pred"]
                    s[f"best_{factor}_heldout_true_prob"] = best[f"{factor}_heldout_true_prob"]

    for s in summary_by_candidate.values():
        if args.cross:
            s["basic_stable"] = bool(s.get("minilm_basic_stable") and s.get("mpnet_basic_stable"))
            s["strong_stable"] = bool(s.get("minilm_strong_stable") and s.get("mpnet_strong_stable"))
            s["cross_stable"] = s["basic_stable"] or s["strong_stable"]
        else:
            s["basic_stable"] = bool(s.get("minilm_basic_stable"))
            s["strong_stable"] = bool(s.get("minilm_strong_stable"))
            s["cross_stable"] = False

    summary_rows = sorted(summary_by_candidate.values(), key=rank_key, reverse=True)

    detail_path = out_dir / "candidate_scores_by_dim.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = out_dir / "candidate_scores_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    top_path = out_dir / "candidate_scores_top.json"
    top_path.write_text(json.dumps(summary_rows[: args.top_k], indent=2), encoding="utf-8")

    print(f"\nSaved {detail_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {top_path}")
    print("\nTop candidates:")
    for s in summary_rows[: args.top_k]:
        print(
            f"{s['candidate_id']:<12} best={s['best_model']}/pca{s['best_dim']} "
            f"LOO={s['best_phase_loo_correct']:>2}/12,"
            f"{s['best_grip_loo_correct']:>2}/12,"
            f"{s['best_hand_loo_correct']:>2}/12 "
            f"basic_total={s['basic_pass_dims_total']} strong_total={s['strong_pass_dims_total']} "
            f"stable_basic={s['basic_stable']} stable_strong={s['strong_stable']}"
        )

    print("\nCounts:")
    print(f"  basic stable: {sum(bool(s['basic_stable']) for s in summary_rows)} / {len(summary_rows)}")
    print(f"  strong stable: {sum(bool(s['strong_stable']) for s in summary_rows)} / {len(summary_rows)}")
    if args.cross:
        print(f"  cross stable: {sum(bool(s['cross_stable']) for s in summary_rows)} / {len(summary_rows)}")

    best = summary_rows[0]
    best_emb = embeddings_by_model[(best["best_model"], best["candidate_id"])]
    best_projected, _ = score_embedding(best_emb, int(best["best_dim"]))
    np.save(out_dir / f"condition_vectors_{best['candidate_id']}_{best['best_model']}_pca{best['best_dim']}.npy", best_projected)
    np.save(out_dir / f"condition_keys_{best['candidate_id']}_{best['best_model']}_pca{best['best_dim']}.npy", KEY_ORDER)
    best_candidate = next(c for c in candidates if c["candidate_id"] == best["candidate_id"])
    (out_dir / "best_sentences.txt").write_text(
        "\n".join(best_candidate["sentences"]) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
