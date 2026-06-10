"""
Run classical baseline classification across many task definitions.

This script:
1) Loads a task from `ClassificationTask` (already prepared in `classification_tasks.py`).
2) Converts each trial signal into compact features (PSD-like mean absolute amplitude).
3) Trains a Logistic Regression model with repeated stratified train/test splits.
4) Computes macro metrics (accuracy, precision, recall, f1).
5) Repeats this for many task definitions and phases.
6) Saves partial and final results to `.npy` files.
7) Writes both console output and a timestamped `.log` file.
"""

#%%
from baseline_linear_classifier.classification_tasks import performance_params, ClassificationTask
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import traceback
from datetime import datetime

# Silence warnings to keep long-run logs clean.
# If you want debugging-level verbosity, remove these 4 lines.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Runtime knobs (environment variables)
# These let you run lighter/faster experiments without editing code.
# Example:
# CLASSIF_N_REP=2 CLASSIF_MAX_ITER=600 CLASSIF_MAX_SAMPLES=4000 python data_classification.py
N_REP = int(os.getenv("CLASSIF_N_REP", "5"))
MAX_ITER = int(os.getenv("CLASSIF_MAX_ITER", "2000"))
TEST_SIZE = float(os.getenv("CLASSIF_TEST_SIZE", "0.25"))
MAX_SAMPLES = int(os.getenv("CLASSIF_MAX_SAMPLES", "6000"))
CONFUSION_ROOT = None

def sanitize_name(name):
    """Convert task names into filesystem-friendly filenames."""
    return (
        str(name)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("=", "_")
    )

class Tee:
    """
    Tiny helper that duplicates output to multiple streams at once.

    We use it to print to terminal and log file simultaneously.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

def format_scores_text(scores):
    """
    Build a compact plain-text summary for easy later lookup.

    The returned text is meant to be human-readable in a `.txt` file
    without requiring NumPy to inspect saved arrays.
    """
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    phase_names = ["PREREACH", "REACH", "GRASP"]
    lines = []
    arr = np.array(scores, dtype=object)
    lines.append("CrossTaskClassification classification summary")
    lines.append("")
    for phase_idx, phase_scores in enumerate(arr):
        lines.append(f"[{phase_names[phase_idx]}]")
        for task_idx, metric_values in enumerate(phase_scores):
            metric_text = ", ".join(
                f"{name}={float(value):.6f}"
                for name, value in zip(metric_names, metric_values)
            )
            lines.append(f"task_{task_idx + 1}: {metric_text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

def save_text_outputs(scores, log_dir, timestamp, log_path):
    """
    Save human-friendly text outputs next to the timestamped log.

    Files written:
    - timestamped summary text file
    - stable `latest` text file for quick lookup
    """
    summary_text = format_scores_text(scores)
    timestamped_txt_path = os.path.join(log_dir, f"data_classification_{timestamp}.txt")
    latest_txt_path = os.path.join(log_dir, "data_classification_latest.txt")
    footer = (
        f"\nSaved log: {log_path}\n"
        f"Saved scores array: scores_all.npy\n"
        f"Saved checkpoint array: scores_all_partial.npy\n"
    )
    full_text = summary_text + footer
    for path in (timestamped_txt_path, latest_txt_path):
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(full_text)
    return timestamped_txt_path, latest_txt_path

def psd_feature_extract(data):
    """
    Convert raw signal tensors into compact per-channel features.

    Expected raw shape per sample is usually:
    (phase, channels, time) OR after task selection (channels, time)

    For arrays with >=3 dims at this stage, we reduce the last axis (time):
    feature[channel] = mean(abs(signal[channel, :]))

    If data is already reduced (ndim < 3), return as is.
    """
    if data.ndim < 3:
        return data
    abs_channel = np.abs(data)
    res = np.mean(abs_channel, axis=2)
    return res

def save_confusion_outputs(cnf_matrix, class_names, output_dir, phase_name, task_name):
    """
    Save confusion-matrix diagnostics for one task.

    Files written:
    - raw count matrix as CSV
    - row-normalized matrix as CSV
    - heatmap PNG using the row-normalized matrix
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_stem = f"{sanitize_name(phase_name)}__{sanitize_name(task_name)}"
    raw_csv_path = os.path.join(output_dir, f"{safe_stem}_counts.csv")
    norm_csv_path = os.path.join(output_dir, f"{safe_stem}_row_normalized.csv")
    png_path = os.path.join(output_dir, f"{safe_stem}.png")

    cnf_matrix = np.asarray(cnf_matrix, dtype=np.int64)
    row_sums = cnf_matrix.sum(axis=1, keepdims=True).astype(np.float64)
    norm_matrix = np.divide(
        cnf_matrix,
        row_sums,
        out=np.zeros_like(cnf_matrix, dtype=np.float64),
        where=row_sums != 0,
    )

    raw_header = "label," + ",".join(class_names)
    raw_rows = [
        ",".join([class_name] + [str(int(v)) for v in row])
        for class_name, row in zip(class_names, cnf_matrix)
    ]
    with open(raw_csv_path, "w", encoding="utf-8") as fp:
        fp.write(raw_header + "\n")
        fp.write("\n".join(raw_rows) + "\n")

    norm_header = "label," + ",".join(class_names)
    norm_rows = [
        ",".join([class_name] + [f"{float(v):.6f}" for v in row])
        for class_name, row in zip(class_names, norm_matrix)
    ]
    with open(norm_csv_path, "w", encoding="utf-8") as fp:
        fp.write(norm_header + "\n")
        fp.write("\n".join(norm_rows) + "\n")

    fig_size = max(5.0, 1.2 * len(class_names))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(norm_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{phase_name} | {task_name}")

    for row_idx in range(cnf_matrix.shape[0]):
        for col_idx in range(cnf_matrix.shape[1]):
            count_value = int(cnf_matrix[row_idx, col_idx])
            norm_value = float(norm_matrix[row_idx, col_idx])
            text_color = "white" if norm_value >= 0.5 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{count_value}\n{norm_value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return raw_csv_path, norm_csv_path, png_path

class RFE_pipeline(Pipeline):
    """
    Pipeline subclass that exposes `coef_` from the final estimator.

    This is useful when a downstream method expects a model with `coef_`
    (historically for feature selection workflows like RFE).
    """
    def fit(self, X, y=None, **fit_params):
        """Fit as normal, then mirror classifier coefficients at pipeline level."""
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self

def test_classification(data_loader, task_name=None, phase_name=None, confusion_dir=None):
    """
    Train/evaluate one task definition and return mean metrics.

    Inputs expected from `data_loader`:
    - data_loader.X : samples for one task
    - data_loader.y : class labels for those samples
    - data_loader.classes : class id -> class name map

    Returns:
    np.array([accuracy, precision_macro, recall_macro, f1_macro])
    averaged over repeated stratified splits.
    """
    # Feature matrix + labels
    X = psd_feature_extract(data_loader.X).astype(np.float32, copy=False)
    y = data_loader.y

    # Optional stratified downsampling to prevent OOM on very large tasks.
    # This preserves class proportions while reducing total samples.
    if MAX_SAMPLES > 0 and y.shape[0] > MAX_SAMPLES:
        subset_split = StratifiedShuffleSplit(n_splits=1, train_size=MAX_SAMPLES, random_state=42)
        subset_idx, _ = next(subset_split.split(X, y))
        X = X[subset_idx, :]
        y = y[subset_idx]
        print(f"[STATUS] DOWNSAMPLED_TO={MAX_SAMPLES}")

    # Repeated random train/test splits with class-balance preservation.
    n_rep = N_REP
    cv_schem = StratifiedShuffleSplit(n_splits=n_rep, test_size=TEST_SIZE, random_state=42)


    # Standard baseline model:
    # - StandardScaler: normalize feature ranges
    # - LogisticRegression: linear multi-class classifier
    # C is tied to number of classes here (project-specific heuristic).
    c_MLR = RFE_pipeline([
        ('std_scal', StandardScaler()),
        ('clf', LogisticRegression(
            C=len(data_loader.classes.keys()),
            penalty='l2',
            solver='lbfgs',
            max_iter=MAX_ITER,
        ))
    ])

    scores = []
    all_true = []
    all_pred = []
    class_ids = sorted(data_loader.classes.keys())
    class_names = [str(data_loader.classes[idx]) for idx in class_ids]

    # Train/evaluate over repeated splits and collect metrics.
    for ind_train, ind_test in cv_schem.split(X, y):
        c_MLR.fit(X[ind_train, :], y[ind_train])
        pred = c_MLR.predict(X[ind_test, :])
        all_true.append(y[ind_test])
        all_pred.append(pred)

        # Macro metrics = equal weight to each class.
        # zero_division=0 avoids crashes when a class has no predicted positives.
        acc = accuracy_score(y[ind_test], pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y[ind_test],
            pred,
            average='macro',
            zero_division=0
        )
        scores.append([
            acc,
            precision,
            recall,
            f1_score
            ])

    if confusion_dir and task_name and phase_name:
        cnf_matrix = confusion_matrix(
            np.concatenate(all_true),
            np.concatenate(all_pred),
            labels=class_ids,
        )
        raw_csv_path, norm_csv_path, png_path = save_confusion_outputs(
            cnf_matrix=cnf_matrix,
            class_names=class_names,
            output_dir=confusion_dir,
            phase_name=phase_name,
            task_name=task_name,
        )
        print(f"[STATUS] CONFUSION_COUNTS_CSV={raw_csv_path}")
        print(f"[STATUS] CONFUSION_NORMALIZED_CSV={norm_csv_path}")
        print(f"[STATUS] CONFUSION_PNG={png_path}")

    # Mean metric vector over all repetitions.
    return np.mean(np.array(scores), axis=0)

def evaluate_task(scores_block, test, task_name, phase_name, task_fn):
    """Run one task, log metrics, and save a confusion matrix."""
    task_fn()
    temp = test_classification(
        test,
        task_name=task_name,
        phase_name=phase_name,
        confusion_dir=CONFUSION_ROOT,
    )
    print(f"{task_name}\n\tFinal score", temp)
    scores_block.append(temp)

def run_all_tasks():
    """
    Execute the full benchmark suite across:
    - 3 phases (prereach/reach/grasp)
    - multiple task formulations (power vs precision, left vs right, angles, etc.)
    - plus global phase-classification tasks at the end

    Saves:
    - scores_all_partial.npy repeatedly (checkpoint)
    - scores_all.npy at completion
    """
    test = ClassificationTask()
    phases = ["PREREACH", "REACH", "GRASP"]
    # scores[phase_index] -> list of metric vectors for each task in that phase
    scores = [[] for _ in range(3)]
    for i in range(3):
        print("="*100)
        print("PHASE",phases[i])
        print("="*100)
        evaluate_task(scores[i], test, "get_task_power_precision", phases[i], lambda: test.get_task_power_precision(phase=i))
        evaluate_task(scores[i], test, "get_task_power_precision_hand(hand=L)", phases[i], lambda: test.get_task_power_precision_hand(hand="L", phase=i))
        evaluate_task(scores[i], test, "get_task_power_precision_hand(hand=R)", phases[i], lambda: test.get_task_power_precision_hand(hand="R", phase=i))
        evaluate_task(scores[i], test, "get_task_power_precision_nobi", phases[i], lambda: test.get_task_power_precision_nobi(phase=i))
        evaluate_task(scores[i], test, "get_task_angles_bimanual", phases[i], lambda: test.get_task_angles_bimanual(phase=i))
        evaluate_task(scores[i], test, "get_task_left_right", phases[i], lambda: test.get_task_left_right(phase=i))
        evaluate_task(scores[i], test, "get_task_left_right_precision", phases[i], lambda: test.get_task_left_right_precision(phase=i))
        evaluate_task(scores[i], test, "get_task_left_right_power", phases[i], lambda: test.get_task_left_right_power(phase=i))
        evaluate_task(scores[i], test, "get_task_angles_hand(hand=L, pp=ALL)", phases[i], lambda: test.get_task_angles_hand(hand="L", pp="ALL", phase=i))
        evaluate_task(scores[i], test, "get_task_angles_hand(hand=L, pp=PRECISION)", phases[i], lambda: test.get_task_angles_hand(hand="L", pp="PRECISION", phase=i))
        evaluate_task(scores[i], test, "get_task_angles_hand(hand=L, pp=POWER)", phases[i], lambda: test.get_task_angles_hand(hand="L", pp="POWER", phase=i))
        evaluate_task(scores[i], test, "get_task_angles_hand(hand=R, pp=ALL)", phases[i], lambda: test.get_task_angles_hand(hand="R", pp="ALL", phase=i))
        evaluate_task(scores[i], test, "get_task_angles_hand(hand=R, pp=PRECISION)", phases[i], lambda: test.get_task_angles_hand(hand="R", pp="PRECISION", phase=i))
        evaluate_task(scores[i], test, "get_task_angles_hand(hand=R, pp=POWER)", phases[i], lambda: test.get_task_angles_hand(hand="R", pp="POWER", phase=i))
        evaluate_task(scores[i], test, "get_task_angles_any_hand(pp=ALL)", phases[i], lambda: test.get_task_angles_any_hand(pp="ALL", phase=i))
        evaluate_task(scores[i], test, "get_task_angles_any_hand(pp=PRECISION)", phases[i], lambda: test.get_task_angles_any_hand(pp="PRECISION", phase=i))
        evaluate_task(scores[i], test, "get_task_angles_any_hand(pp=POWER)", phases[i], lambda: test.get_task_angles_any_hand(pp="POWER", phase=i))
        evaluate_task(scores[i], test, "get_task_unimanual_bimanual", phases[i], lambda: test.get_task_unimanual_bimanual(phase=i))

        # Checkpoint after each phase to avoid losing progress on abrupt termination.
        np.save("scores_all_partial", np.array(scores, dtype=object))

    # Additional task family: classify phase itself (across all samples).
    evaluate_task(scores[0], test, "get_task_phases(pp=ALL)", "ALL_PHASES", lambda: test.get_task_phases(pp="ALL"))
    np.save("scores_all_partial", np.array(scores, dtype=object))
    
    evaluate_task(scores[1], test, "get_task_phases(pp=PRECISION)", "ALL_PHASES", lambda: test.get_task_phases(pp="PRECISION"))
    np.save("scores_all_partial", np.array(scores, dtype=object))

    evaluate_task(scores[2], test, "get_task_phases(pp=POWER)", "ALL_PHASES", lambda: test.get_task_phases(pp="POWER"))
    np.save("scores_all_partial", np.array(scores, dtype=object))
    print(np.array(scores).shape)
    final_scores = np.array(scores)
    np.save("scores_all", final_scores)
    return final_scores

#%%
if __name__ == "__main__":
    # Exit status convention:
    # 0   -> success
    # 1   -> unhandled Python exception
    # 130 -> keyboard interrupt (Ctrl+C)
    exit_code = 0

    # Build timestamped log path near this script.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"data_classification_{timestamp}.log")
    confusion_root = os.path.join(log_dir, "data_classification_confusion_matrices", timestamp)
    os.makedirs(confusion_root, exist_ok=True)

    # Mirror stdout/stderr to both terminal and logfile.
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        print(f"[STATUS] LOG_FILE={log_path}")
        print(f"[STATUS] CONFUSION_DIR={confusion_root}")
        print(f"[STATUS] START {datetime.now().isoformat(timespec='seconds')}")
        try:
            CONFUSION_ROOT = confusion_root
            final_scores = run_all_tasks()
            summary_path, latest_path = save_text_outputs(
                final_scores,
                log_dir=log_dir,
                timestamp=timestamp,
                log_path=log_path,
            )
            print(f"[STATUS] SUMMARY_TXT={summary_path}")
            print(f"[STATUS] LATEST_TXT={latest_path}")
        except KeyboardInterrupt:
            exit_code = 130
            print("[ERROR] KeyboardInterrupt")
            traceback.print_exc()
        except Exception as exc:
            # Print full traceback to simplify debugging from logs.
            exit_code = 1
            print(f"[ERROR] {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            print(f"[STATUS] EXIT_CODE={exit_code}")
            # Restore default streams before process exit.
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    sys.exit(exit_code)
