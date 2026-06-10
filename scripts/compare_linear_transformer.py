"""Compare transformer and linear-baseline phase/grip/hand outputs.

The transformer and linear baseline save similar information in different file
layouts. This utility reads both summaries and writes one comparable table plus
raw confusion matrices in a common JSON format.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

HEADS = ("phase", "grip", "hand")
SPLITS = ("seen_test", "heldout_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare transformer vs linear baseline outputs.")
    p.add_argument("--transformer_dir", type=Path, required=True)
    p.add_argument("--linear_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def transformer_result(summary: dict[str, Any], split: str, head: str) -> dict[str, Any]:
    item = summary[split][head]
    cm = np.asarray(item["confusion_matrix"], dtype=int)
    return {
        "accuracy": float(item["accuracy"]),
        "macro_f1": macro_f1_from_confusion(cm),
        "confusion_matrix": cm,
    }


def linear_result(summary: dict[str, Any], split: str, head: str) -> dict[str, Any]:
    item = summary["heads"][head][split]
    cm = np.asarray(item["confusion_matrix"], dtype=int)
    return {
        "accuracy": float(item["accuracy"]),
        "macro_f1": float(item.get("macro_f1", macro_f1_from_confusion(cm))),
        "confusion_matrix": cm,
    }


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Compute macro-F1 from a confusion matrix, including absent-label rows as 0."""
    f1s = []
    for k in range(cm.shape[0]):
        tp = float(cm[k, k])
        fp = float(cm[:, k].sum() - cm[k, k])
        fn = float(cm[k, :].sum() - cm[k, k])
        if tp == 0.0 and fp == 0.0 and fn == 0.0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def row_normalized(cm: np.ndarray) -> list[list[float]]:
    denom = cm.sum(axis=1, keepdims=True).astype(float)
    norm = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
    return norm.tolist()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.linear_dir.parent / "comparison_linear_vs_transformer")
    out_dir.mkdir(parents=True, exist_ok=True)

    transformer = load_json(args.transformer_dir / "summary.json")
    linear = load_json(args.linear_dir / "summary.json")

    rows = []
    matrices: dict[str, Any] = {}
    for split in SPLITS:
        matrices[split] = {}
        for head in HEADS:
            t = transformer_result(transformer, split, head)
            l = linear_result(linear, split, head)
            rows.append({
                "split": split,
                "head": head,
                "transformer_accuracy": t["accuracy"],
                "linear_accuracy": l["accuracy"],
                "linear_minus_transformer_accuracy": l["accuracy"] - t["accuracy"],
                "transformer_macro_f1_from_cm": t["macro_f1"],
                "linear_macro_f1_from_cm": l["macro_f1"],
                "linear_minus_transformer_macro_f1": l["macro_f1"] - t["macro_f1"],
            })
            matrices[split][head] = {
                "transformer_counts": t["confusion_matrix"].tolist(),
                "linear_counts": l["confusion_matrix"].tolist(),
                "transformer_row_normalized": row_normalized(t["confusion_matrix"]),
                "linear_row_normalized": row_normalized(l["confusion_matrix"]),
            }

    csv_path = out_dir / "comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "comparison_confusion_matrices.json"
    json_path.write_text(json.dumps(matrices, indent=2), encoding="utf-8")

    md_lines = ["# Linear Baseline vs Transformer", ""]
    md_lines.append("| split | head | transformer acc | linear acc | linear - transformer | transformer macro-F1 | linear macro-F1 |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md_lines.append(
            f"| {r['split']} | {r['head']} | "
            f"{r['transformer_accuracy']:.4f} | {r['linear_accuracy']:.4f} | "
            f"{r['linear_minus_transformer_accuracy']:+.4f} | "
            f"{r['transformer_macro_f1_from_cm']:.4f} | {r['linear_macro_f1_from_cm']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("Note: heldout_test contains only the held-out phase/grip/hand labels, so its confusion matrices have empty rows for labels that are absent by design.")
    md_path = out_dir / "comparison_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("\n" + md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
