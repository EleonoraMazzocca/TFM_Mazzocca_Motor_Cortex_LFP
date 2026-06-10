#!/usr/bin/env python3
"""
Static leakage verification for Manuel's classification pipeline.

This tool inspects source code only (no execution of pipeline code) and writes
a plain-text report with per-check verdicts:
- LEAKAGE
- POSSIBLE
- CLEAN
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


VERDICTS = ("LEAKAGE", "POSSIBLE", "CLEAN")


@dataclass
class Evidence:
    path: Path
    line_no: int
    text: str


@dataclass
class CheckResult:
    name: str
    finding: str
    verdict: str
    evidence: List[Evidence]
    manual_note: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static leakage verification for CrossTaskClassification scripts."
    )
    parser.add_argument(
        "--classification",
        type=Path,
        required=True,
        help="Path to data_classification.py",
    )
    parser.add_argument(
        "--standardization",
        type=Path,
        required=True,
        help="Path to data_standardization.py",
    )
    parser.add_argument(
        "--preprocess",
        type=Path,
        required=True,
        help="Path to data_preprocess.py",
    )
    parser.add_argument(
        "--extra-scripts",
        type=Path,
        nargs="*",
        default=[],
        help="Any other preprocessing / pipeline scripts to inspect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output report path. If omitted, prints to stdout only.",
    )
    return parser.parse_args()


def load_sources(paths: Sequence[Path]) -> Dict[Path, List[str]]:
    sources: Dict[Path, List[str]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        sources[path] = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return sources


def find_evidence(
    sources: Dict[Path, List[str]],
    patterns: Iterable[str],
    *,
    path_filter: Iterable[Path] | None = None,
    max_hits: int = 8,
) -> List[Evidence]:
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    allowed = set(path_filter) if path_filter is not None else None
    out: List[Evidence] = []

    for path, lines in sources.items():
        if allowed is not None and path not in allowed:
            continue
        for idx, line in enumerate(lines, start=1):
            if any(rx.search(line) for rx in compiled):
                out.append(Evidence(path=path, line_no=idx, text=line.rstrip()))
                if len(out) >= max_hits:
                    return out
    return out


def has_pattern_in_path(
    sources: Dict[Path, List[str]],
    path: Path,
    pattern: str,
) -> bool:
    rx = re.compile(pattern, flags=re.IGNORECASE)
    return any(rx.search(line) for line in sources[path])


def build_check_1(
    sources: Dict[Path, List[str]],
    classification: Path,
    extra_paths: Sequence[Path],
) -> CheckResult:
    split_ev = find_evidence(
        sources,
        [r"StratifiedShuffleSplit", r"cv_schem\.split\(", r"test_size\s*="],
        path_filter=[classification],
        max_hits=6,
    )
    group_ev = find_evidence(
        sources,
        [r"GroupKFold", r"GroupShuffleSplit", r"groups\s*="],
        path_filter=[classification],
        max_hits=3,
    )
    mixed_session_ev = find_evidence(
        sources,
        [
            r"for file_name in data_files",
            r"\.append\(data\[i\]\)",
            r"np\.save\(SEPARATED_CLASSES_DIR",
            r"self\.X = np\.concatenate\(self\.X\)",
        ],
        max_hits=8,
    )

    split_is_trial_level = bool(split_ev)
    has_grouping = bool(group_ev)
    session_mixing_evidence = len(mixed_session_ev) >= 2

    if split_is_trial_level and not has_grouping and session_mixing_evidence:
        verdict = "LEAKAGE"
        finding = (
            "The classifier uses repeated stratified random trial-level splits, and the upstream "
            "pipeline merges trials across sessions into class arrays without preserving session IDs. "
            "So same-session trials can enter both train and test."
        )
    elif split_is_trial_level and not has_grouping:
        verdict = "POSSIBLE"
        finding = (
            "The classifier uses trial-level random splits with no explicit session grouping. "
            "Static code does not prove same-session overlap, but it is possible."
        )
    else:
        verdict = "CLEAN"
        finding = "A session/group-aware split strategy appears to be in use."

    manual_note = None
    session_split_helper = [p for p in extra_paths if p.name == "build_session_aware_structured_split.py"]
    if session_split_helper:
        manual_note = (
            "A session-aware split helper script exists, but static analysis cannot confirm it was used "
            "for the reported Classifier 1 results. Verify run logs / manifests."
        )

    return CheckResult(
        name="Check 1 — Split strategy",
        finding=finding,
        verdict=verdict,
        evidence=split_ev + mixed_session_ev[: max(0, 8 - len(split_ev))],
        manual_note=manual_note,
    )


def build_check_2(
    sources: Dict[Path, List[str]],
    classification: Path,
) -> CheckResult:
    scaler_ev = find_evidence(
        sources,
        [r"StandardScaler", r"\('std_scal',\s*StandardScaler\(", r"c_MLR\.fit\(X\[ind_train"],
        path_filter=[classification],
        max_hits=8,
    )

    full_fit_ev = find_evidence(
        sources,
        [r"\.fit\(\s*X\s*,\s*y\s*\)", r"fit_transform\(\s*X\s*\)"],
        path_filter=[classification],
        max_hits=4,
    )
    has_explicit_full_fit = any("ind_train" not in ev.text for ev in full_fit_ev)

    if scaler_ev and not has_explicit_full_fit:
        verdict = "CLEAN"
        finding = (
            "Standardization is inside a sklearn Pipeline and the model is fit on train indices "
            "within each split, so scaler statistics come from training data only."
        )
    elif scaler_ev:
        verdict = "POSSIBLE"
        finding = (
            "StandardScaler is present, but static pattern matching found potential full-dataset fit "
            "calls. Manual confirmation is needed."
        )
    else:
        verdict = "POSSIBLE"
        finding = (
            "No explicit split-aware standardization pattern was detected in the classifier script."
        )

    manual_note = None
    if verdict != "CLEAN":
        manual_note = (
            "Inspect runtime objects or debug logs to confirm scaler.fit receives training-only rows."
        )

    return CheckResult(
        name="Check 2 — Standardization leakage",
        finding=finding,
        verdict=verdict,
        evidence=scaler_ev + full_fit_ev[: max(0, 8 - len(scaler_ev))],
        manual_note=manual_note,
    )


def build_check_3(
    sources: Dict[Path, List[str]],
    classification: Path,
    preprocess: Path,
) -> CheckResult:
    pca_ev = find_evidence(
        sources,
        [r"\bPCA\s*\(", r"sklearn\.decomposition", r"\.fit\(.+PCA"],
        max_hits=5,
    )
    feature_ev = find_evidence(
        sources,
        [r"X\s*=\s*psd_feature_extract\(data_loader\.X\)", r"np\.mean\(np\.abs\("],
        path_filter=[classification],
        max_hits=5,
    )
    bad_channel_ev = find_evidence(
        sources,
        [r"std_bad_channels", r"std_all\s*=\s*channelwise_std\(lfp\)", r"lfp\d?_data\[bad_channels"],
        path_filter=[preprocess],
        max_hits=6,
    )
    bandpass_ev = find_evidence(
        sources,
        [r"bandpass_filtering", r"iirfilter", r"filtfilt"],
        path_filter=[preprocess],
        max_hits=3,
    )

    if pca_ev:
        verdict = "LEAKAGE"
        finding = (
            "PCA-related fitting was detected in inspected scripts; if fit before split this is leakage."
        )
    elif bad_channel_ev:
        verdict = "POSSIBLE"
        finding = (
            "No PCA leakage pattern detected. Feature extraction appears per-sample (mean abs amplitude), "
            "but data-dependent bad-channel statistics are computed on full recordings before trial split, "
            "which can leak test-distribution information."
        )
    else:
        verdict = "CLEAN"
        finding = (
            "No data-dependent global feature transform (like PCA fit on full data) was detected; "
            "fixed filtering and per-sample feature extraction are not leakage by themselves."
        )

    manual_note = None
    if verdict == "POSSIBLE":
        manual_note = (
            "To rule this out, recompute bad-channel masks using training sessions only and compare metrics."
        )

    evidence = pca_ev + feature_ev + bad_channel_ev + bandpass_ev
    return CheckResult(
        name="Check 3 — Feature extraction leakage",
        finding=finding,
        verdict=verdict,
        evidence=evidence[:8],
        manual_note=manual_note,
    )


def build_check_4(
    sources: Dict[Path, List[str]],
    classification: Path,
    task_file: Path | None,
) -> CheckResult:
    task_ev: List[Evidence] = []
    if task_file is not None:
        task_ev = find_evidence(
            sources,
            [
                r"def get_task_phases",
                r"phase0\s*=\s*np\.array\(data\[:,\s*0\]",
                r"phase1\s*=\s*np\.array\(data\[:,\s*1\]",
                r"phase2\s*=\s*np\.array\(data\[:,\s*2\]",
            ],
            path_filter=[task_file],
            max_hits=6,
        )

    split_ev = find_evidence(
        sources,
        [
            r"StratifiedShuffleSplit",
            r"for ind_train, ind_test in cv_schem\.split\(X,\s*y\)",
            r"get_task_phases\(pp=",
        ],
        path_filter=[classification],
        max_hits=6,
    )

    has_phase_stack = False
    has_phase_samples = False
    if task_file is not None:
        has_phase_stack = has_pattern_in_path(sources, task_file, r"def get_task_phases")
        has_phase_samples = (
            has_pattern_in_path(sources, task_file, r"phase0\s*=\s*np\.array\(data\[:,\s*0\]")
            and has_pattern_in_path(sources, task_file, r"phase1\s*=\s*np\.array\(data\[:,\s*1\]")
            and has_pattern_in_path(sources, task_file, r"phase2\s*=\s*np\.array\(data\[:,\s*2\]")
        )

    has_phase_task_called = has_pattern_in_path(sources, classification, r"get_task_phases\(pp=")
    has_random_split = (
        has_pattern_in_path(sources, classification, r"StratifiedShuffleSplit")
        and has_pattern_in_path(sources, classification, r"cv_schem\.split\(X,\s*y\)")
    )

    if has_phase_stack and has_phase_samples and has_phase_task_called and has_random_split:
        verdict = "LEAKAGE"
        finding = (
            "For phase-classification tasks, the three phases of each original trial are converted into "
            "separate samples and then randomly split, so sibling samples from the same trial can appear "
            "in both train and test."
        )
    elif has_phase_stack and has_phase_task_called:
        verdict = "POSSIBLE"
        finding = (
            "Phase samples are built from the same original trials, but split behavior could not be fully verified."
        )
    else:
        verdict = "POSSIBLE"
        finding = (
            "Could not confirm whether per-trial phases are grouped during splitting from static inspection."
        )

    manual_note = None
    if verdict != "CLEAN":
        manual_note = (
            "Use a grouped split keyed by (session, trial_index) so all phases of a trial stay together."
        )

    evidence = task_ev + split_ev
    return CheckResult(
        name="Check 4 — Trial independence",
        finding=finding,
        verdict=verdict,
        evidence=evidence[:8],
        manual_note=manual_note,
    )


def build_check_5(
    sources: Dict[Path, List[str]],
    classification: Path,
    preprocess: Path,
    task_file: Path | None,
    extra_paths: Sequence[Path],
) -> CheckResult:
    label_ev = find_evidence(
        sources,
        [
            r'if "PRECISION" in l',
            r'elif "POWER" in l',
            r"motor_no",
            r"angle",
            r"dc_info\[\"Precision/Power\"\]",
        ],
        path_filter=[p for p in [task_file, preprocess] if p is not None],
        max_hits=8,
    )
    heldout_ev = find_evidence(
        sources,
        [r"held-?out", r"holdout", r"compositional"],
        path_filter=[classification, *extra_paths],
        max_hits=6,
    )
    uses_holdout_in_classifier = has_pattern_in_path(sources, classification, r"held-?out|holdout")

    if label_ev and not uses_holdout_in_classifier:
        verdict = "POSSIBLE"
        finding = (
            "Label construction appears metadata/file-name based (not derived from neural signal values), "
            "which is clean for label leakage. However, the classifier script does not define a fixed held-out "
            "test set, so post-hoc test-set definition cannot be ruled out from static code alone."
        )
        manual_note = (
            "Verify experiment protocol or logs to confirm any final test set was decided before model tuning."
        )
    elif label_ev:
        verdict = "CLEAN"
        finding = (
            "Labels appear to come from external task metadata/filenames, and a held-out rule is explicitly defined."
        )
        manual_note = None
    else:
        verdict = "POSSIBLE"
        finding = "Could not reliably verify label origin and held-out policy from static analysis."
        manual_note = (
            "Inspect label-generation scripts and experiment logs for any target encoding from signal-derived fields."
        )

    return CheckResult(
        name="Check 5 — Label leakage",
        finding=finding,
        verdict=verdict,
        evidence=(label_ev + heldout_ev)[:8],
        manual_note=manual_note,
    )


def overall_verdict(results: Sequence[CheckResult]) -> str:
    verdicts = [r.verdict for r in results]
    if "LEAKAGE" in verdicts:
        return "LEAKAGE"
    if "POSSIBLE" in verdicts:
        return "POSSIBLE"
    return "CLEAN"


def format_evidence_block(evidence: Sequence[Evidence], limit: int = 6) -> str:
    if not evidence:
        return "  Evidence: (no direct line match found)\n"
    lines = ["  Evidence:"]
    for ev in evidence[:limit]:
        quoted = ev.text.strip()
        lines.append(f'    - "{quoted}" ({ev.path.name}:{ev.line_no})')
    return "\n".join(lines) + "\n"


def build_summary(overall: str, results: Sequence[CheckResult]) -> str:
    if overall == "LEAKAGE":
        return (
            "Static inspection indicates Classifier 1 results are likely optimistic because at least one "
            "split-independence rule is violated in code (notably trial/phase or session leakage paths). "
            "These results should not be treated as a strong thesis baseline until re-evaluated with "
            "grouped/session-aware splits and strict trial-level independence."
        )
    if overall == "POSSIBLE":
        return (
            "No single definitive leakage bug is proven across all checks, but unresolved risks remain "
            "from split design and/or preprocessing scope. Classifier 1 can be used only as a provisional "
            "baseline, with explicit caveats until manual protocol checks and stricter split controls are added."
        )
    return (
        "No leakage was detected by static code inspection for the requested checks. Classifier 1 results can "
        "be considered a baseline, but should still be documented with split protocol details and reproducibility logs."
    )


def main() -> None:
    args = parse_args()

    all_paths: List[Path] = [args.classification, args.standardization, args.preprocess, *args.extra_scripts]
    # Remove duplicates while preserving order.
    dedup_paths: List[Path] = []
    seen = set()
    for p in all_paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            dedup_paths.append(p)

    sources = load_sources(dedup_paths)

    task_file: Path | None = None
    for p in dedup_paths:
        if p.name == "classification_tasks.py":
            task_file = p
            break

    checks: List[CheckResult] = [
        build_check_1(sources, classification=args.classification, extra_paths=args.extra_scripts),
        build_check_2(sources, classification=args.classification),
        build_check_3(sources, classification=args.classification, preprocess=args.preprocess),
        build_check_4(sources, classification=args.classification, task_file=task_file),
        build_check_5(
            sources,
            classification=args.classification,
            preprocess=args.preprocess,
            task_file=task_file,
            extra_paths=args.extra_scripts,
        ),
    ]
    overall = overall_verdict(checks)

    out_lines: List[str] = []
    out_lines.append("LEAKAGE VERIFICATION REPORT")
    out_lines.append("============================")
    out_lines.append("")
    for result in checks:
        out_lines.append(result.name)
        out_lines.append(f"  Finding: {result.finding}")
        out_lines.append(f"  Verdict: {result.verdict}")
        out_lines.append(format_evidence_block(result.evidence).rstrip("\n"))
        if result.manual_note:
            out_lines.append(f"  Manual follow-up: {result.manual_note}")
        out_lines.append("")

    out_lines.append(f"OVERALL VERDICT: {overall}")
    out_lines.append(f"SUMMARY: {build_summary(overall, checks)}")
    report = "\n".join(out_lines).rstrip() + "\n"

    print(report, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
