"""Grid-search sentence templates for soft cVAE condition labels.

The search is deliberately lightweight: it never trains the cVAE.  It creates
candidate sets of 12 condition sentences, embeds them with a sentence model,
projects them to PCA dimensions, and scores whether phase/grip/hand are
recoverable but not trivially separable.

Usage:
    python search_condition_sentences.py --out_dir results/sentence_search
    python search_condition_sentences.py --top_k 20 --dims 2 3 4 5
    python search_condition_sentences.py --candidates_json llm_candidates.json --out_dir results/sentence_search_llm
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np


PHASES = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
GRIPS = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
HANDS = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
KEY_ORDER = np.stack([PHASES, GRIPS, HANDS], axis=1)
HELDOUT_IDX = 11  # grasp + precision + right

PHASE_SETS = {
    "explicit": [
        "preparation before reaching",
        "movement toward the object",
        "object contact and grasp closure",
    ],
    "ordinal": [
        "early part of the reach-to-grasp sequence",
        "middle part of the reach-to-grasp sequence",
        "late part of the reach-to-grasp sequence",
    ],
    "contextual": [
        "quiet readiness before the action unfolds",
        "active approach while the limb travels",
        "final interaction as the object is secured",
    ],
    "abstract": [
        "initial motor context",
        "transitional motor context",
        "terminal motor context",
    ],
}

GRIP_SETS = {
    "explicit": ["power grip", "precision grip"],
    "functional": ["broad stable hold", "delicate controlled contact"],
    "object": ["larger contact surface", "smaller contact surface"],
    "abstract": ["high-force object interaction", "fine-control object interaction"],
}

HAND_SETS = {
    "explicit": ["left hand", "right hand"],
    "side": ["left-side limb", "right-side limb"],
    "spatial": ["one lateral side of the workspace", "the opposite lateral side of the workspace"],
    "abstract": ["first effector side", "second effector side"],
}

TEMPLATES = {
    "comma": "{phase}, {grip}, {hand}",
    "trial": "trial context: {phase}; object interaction: {grip}; effector: {hand}",
    "sentence": "The movement is in {phase}, with {grip}, using {hand}.",
    "holistic": "A motor episode involving {phase} and {grip} on {hand}.",
    "minimal": "{phase}; {grip}; {hand}.",
}


def build_sentences(phase_key: str, grip_key: str, hand_key: str, template_key: str) -> list[str]:
    phases = PHASE_SETS[phase_key]
    grips = GRIP_SETS[grip_key]
    hands = HAND_SETS[hand_key]
    template = TEMPLATES[template_key]
    return [
        template.format(phase=phases[p], grip=grips[g], hand=hands[h])
        for p, g, h in KEY_ORDER
    ]


def iter_candidates(candidates_json: str | None):
    if candidates_json is not None:
        data = json.loads(Path(candidates_json).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--candidates_json must contain a JSON list")
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Candidate {i} must be an object")
            candidate_id = str(item.get("candidate_id", f"external_{i:03d}"))
            sentences = item.get("sentences")
            if not isinstance(sentences, list) or len(sentences) != 12:
                raise ValueError(f"{candidate_id}: expected exactly 12 sentences")
            if not all(isinstance(sentence, str) and sentence.strip() for sentence in sentences):
                raise ValueError(f"{candidate_id}: all sentences must be non-empty strings")
            yield {
                "candidate_id": candidate_id,
                "phase_set": "external",
                "grip_set": "external",
                "hand_set": "external",
                "template": "external",
                "sentences": sentences,
            }
        return

    for phase_key, grip_key, hand_key, template_key in product(
        PHASE_SETS, GRIP_SETS, HAND_SETS, TEMPLATES
    ):
        yield {
            "candidate_id": f"ph-{phase_key}__gr-{grip_key}__ha-{hand_key}__tpl-{template_key}",
            "phase_set": phase_key,
            "grip_set": grip_key,
            "hand_set": hand_key,
            "template": template_key,
            "sentences": build_sentences(phase_key, grip_key, hand_key, template_key),
        }


def factor_scores(x: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    clf = LogisticRegression(max_iter=10000, solver="lbfgs")
    clf.fit(x, labels)
    train_acc = float(clf.score(x, labels))
    loo_acc = float(cross_val_score(clf, x, labels, cv=LeaveOneOut()).mean())

    train_mask = np.arange(len(labels)) != HELDOUT_IDX
    clf.fit(x[train_mask], labels[train_mask])
    heldout_pred = int(clf.predict(x[[HELDOUT_IDX]])[0])
    heldout_true = int(labels[HELDOUT_IDX])
    heldout_proba = clf.predict_proba(x[[HELDOUT_IDX]])[0]
    class_to_col = {int(cls): i for i, cls in enumerate(clf.classes_)}
    heldout_true_prob = float(heldout_proba[class_to_col[heldout_true]])
    heldout_max_prob = float(np.max(heldout_proba))

    return {
        "train_acc": train_acc,
        "loo_acc": loo_acc,
        "heldout_pred": heldout_pred,
        "heldout_true": heldout_true,
        "heldout_correct": bool(heldout_pred == heldout_true),
        "heldout_true_prob": heldout_true_prob,
        "heldout_max_prob": heldout_max_prob,
    }


def score_candidate(emb: np.ndarray, dim: int) -> tuple[np.ndarray, dict]:
    from sklearn.decomposition import PCA

    x = PCA(n_components=dim, random_state=42).fit_transform(emb).astype(np.float32)
    scores = {
        "phase": factor_scores(x, PHASES),
        "grip": factor_scores(x, GRIPS),
        "hand": factor_scores(x, HANDS),
    }

    # We want useful but not symbolic conditions.  Penalize perfect train/LOO
    # separability and held-out certainty, but also penalize falling below chance.
    targets = {"phase": 0.85, "grip": 0.75, "hand": 0.65}
    chance = {"phase": 1.0 / 3.0, "grip": 0.5, "hand": 0.5}
    true_prob_targets = {"phase": 0.70, "grip": 0.60, "hand": 0.55}
    total = 0.0
    for name in ("phase", "grip", "hand"):
        loo = scores[name]["loo_acc"]
        train = scores[name]["train_acc"]
        total -= abs(loo - targets[name])
        total -= 0.5 * max(0.0, loo - 0.95)
        total -= 0.5 * max(0.0, chance[name] - loo)
        total -= 0.25 * max(0.0, train - 0.95)
        total -= 0.25 * max(0.0, scores[name]["heldout_max_prob"] - 0.85)
        total -= 0.25 * abs(scores[name]["heldout_true_prob"] - true_prob_targets[name])
    scores["score"] = float(total)
    return x, scores


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=str, default="results/sentence_search")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument("--candidates_json", type=str, default=None,
                   help="Optional JSON list of {'candidate_id', 'sentences'} entries to score.")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(f"Missing dependency: {e}")

    print(f"Loading {args.model} ...")
    lm = SentenceTransformer(args.model)

    rows = []
    cache = {}
    for candidate in iter_candidates(args.candidates_json):
        candidate_id = candidate["candidate_id"]
        sentences = candidate["sentences"]
        emb = lm.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        cache[candidate_id] = {"sentences": sentences, "emb": emb.astype(np.float32)}
        for dim in args.dims:
            projected, scores = score_candidate(emb, dim)
            row = {
                "candidate_id": candidate_id,
                "phase_set": candidate["phase_set"],
                "grip_set": candidate["grip_set"],
                "hand_set": candidate["hand_set"],
                "template": candidate["template"],
                "dim": dim,
                "score": scores["score"],
            }
            for factor in ("phase", "grip", "hand"):
                for metric, value in scores[factor].items():
                    row[f"{factor}_{metric}"] = value
            rows.append(row)

    rows.sort(key=lambda r: r["score"], reverse=True)
    csv_path = out_dir / "sentence_search_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top = rows[: args.top_k]
    (out_dir / "sentence_search_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")

    best = rows[0]
    best_id = best["candidate_id"]
    best_dim = int(best["dim"])
    best_sentences = cache[best_id]["sentences"]
    best_emb = cache[best_id]["emb"]
    best_projected, _ = score_candidate(best_emb, best_dim)

    np.save(out_dir / f"condition_vectors_best_pca{best_dim}.npy", best_projected)
    np.save(out_dir / f"condition_keys_best_pca{best_dim}.npy", KEY_ORDER)
    (out_dir / "best_sentences.txt").write_text("\n".join(best_sentences) + "\n", encoding="utf-8")

    print(f"\nSaved {csv_path}")
    print(f"Best: {best_id} | PCA dim={best_dim} | score={best['score']:.3f}")
    for factor in ("phase", "grip", "hand"):
        print(
            f"  {factor}: train={best[f'{factor}_train_acc']:.3f} "
            f"loo={best[f'{factor}_loo_acc']:.3f} "
            f"heldout={best[f'{factor}_heldout_pred']}/{best[f'{factor}_heldout_true']}"
        )
    print(f"Saved best condition table to {out_dir}")


if __name__ == "__main__":
    main()
