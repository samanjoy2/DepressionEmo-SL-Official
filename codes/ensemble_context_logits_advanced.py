import os
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


CONTEXT_VALUES = ["C0", "C1", "C2", "C3"]


def find_context_from_name(name: str) -> str:
    parts = name.split("_")
    for p in parts:
        if p in CONTEXT_VALUES:
            return p
    return ""


def build_group_key(name: str, context: str) -> str:
    parts = [p for p in name.split("_") if p != context]
    return "_".join(parts)


def discover_groups(report_dir: str) -> Dict[str, Dict[str, str]]:
    groups: Dict[str, Dict[str, str]] = defaultdict(dict)

    for entry in os.listdir(report_dir):
        full_path = os.path.join(report_dir, entry)
        if not os.path.isdir(full_path):
            continue

        if entry.startswith("[") and entry.endswith("]"):
            continue

        context = find_context_from_name(entry)
        if not context:
            continue

        logits_path = os.path.join(full_path, "test_logits.csv")
        if not os.path.isfile(logits_path):
            continue

        group_key = build_group_key(entry, context)
        groups[group_key][context] = full_path

    return groups


def align_logits_frames(
    context_to_path: Dict[str, str]
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:

    contexts = [c for c in CONTEXT_VALUES if c in context_to_path]
    if not contexts:
        raise ValueError("No valid contexts found in context_to_path.")

    raw_dfs: Dict[str, pd.DataFrame] = {}
    for ctx in contexts:
        path = os.path.join(context_to_path[ctx], "test_logits.csv")
        df = pd.read_csv(path)
        if "example_id" not in df.columns:
            raise ValueError(f"'example_id' column missing in {path}")
        raw_dfs[ctx] = df


    common_ids = None
    for ctx, df in raw_dfs.items():
        ids = set(df["example_id"].astype(str).tolist())
        if common_ids is None:
            common_ids = ids
        else:
            common_ids &= ids

    if not common_ids:
        raise ValueError("No overlapping example_id values across contexts.")

    ordered_ids = sorted(common_ids)


    aligned: Dict[str, pd.DataFrame] = {}
    for ctx, df in raw_dfs.items():
        df = df.copy()
        df["example_id"] = df["example_id"].astype(str)
        df = df[df["example_id"].isin(ordered_ids)]
        df = df.set_index("example_id").loc[ordered_ids].reset_index()
        aligned[ctx] = df


    ref_ctx = contexts[0]
    ref_df = aligned[ref_ctx]
    ref_true_ids = ref_df["true_label_id"].tolist()
    ref_true_labels = ref_df["true_label"].tolist()

    for ctx in contexts[1:]:
        df = aligned[ctx]
        if df["true_label_id"].tolist() != ref_true_ids:
            raise ValueError(
                f"true_label_id mismatch between contexts {ref_ctx} and {ctx}"
            )
        if df["true_label"].tolist() != ref_true_labels:
            raise ValueError(
                f"true_label mismatch between contexts {ref_ctx} and {ctx}"
            )

    return contexts, aligned


def compute_ensemble_predictions(
    contexts: List[str],
    aligned: Dict[str, pd.DataFrame],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, np.ndarray],
]:
    ref_ctx = contexts[0]
    ref_df = aligned[ref_ctx]

    true_label_ids = ref_df["true_label_id"].to_numpy(dtype=int)


    logit_cols = [c for c in ref_df.columns if c.startswith("logit_")]
    if not logit_cols:
        raise ValueError("No logit_* columns found in reference DataFrame.")
    label_names = [c[len("logit_") :] for c in logit_cols]
    num_labels = len(label_names)


    for ctx in contexts[1:]:
        cols = [c for c in aligned[ctx].columns if c.startswith("logit_")]
        if cols != logit_cols:
            raise ValueError(
                f"logit column mismatch between contexts {ref_ctx} and {ctx}"
            )


    per_context_logits: Dict[str, np.ndarray] = {}
    per_context_preds: Dict[str, np.ndarray] = {}
    logits_list: List[np.ndarray] = []

    for ctx in contexts:
        logits = aligned[ctx][logit_cols].to_numpy(dtype=float)
        per_context_logits[ctx] = logits
        per_context_preds[ctx] = logits.argmax(axis=1)
        logits_list.append(logits)

    stacked_logits = np.stack(logits_list, axis=0)
    mean_logits = stacked_logits.mean(axis=0)
    soft_pred_ids = mean_logits.argmax(axis=1)


    votes = np.stack(list(per_context_preds.values()), axis=0)
    num_examples = votes.shape[1]
    hard_pred_ids = np.zeros(num_examples, dtype=int)

    for i in range(num_examples):
        counts = Counter(votes[:, i].tolist())
        max_count = max(counts.values())
        candidates = [cls for cls, cnt in counts.items() if cnt == max_count]
        if len(candidates) == 1:
            hard_pred_ids[i] = candidates[0]
        else:

            candidate_scores = [mean_logits[i, c] for c in candidates]
            best_idx = int(np.argmax(candidate_scores))
            hard_pred_ids[i] = candidates[best_idx]

    return (
        true_label_ids,
        soft_pred_ids,
        hard_pred_ids,
        label_names,
        per_context_preds,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Tuple[float, float, str]:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    return macro_f1, acc, report


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    out_path: str,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt

        num_labels = len(label_names)
        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(num_labels))
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        tick_marks = np.arange(num_labels)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticklabels(label_names)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title(title)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(num_labels):
            for j in range(num_labels):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color=("white" if cm[i, j] > thresh else "black"),
                )
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not save confusion matrix to {out_path}: {e}")


def process_group(
    group_key: str,
    context_to_path: Dict[str, str],
    ensemble_root: str,
) -> None:

    missing = [c for c in CONTEXT_VALUES if c not in context_to_path]
    if missing:
        print(
            f"[SKIP] Group {group_key}: missing contexts {', '.join(missing)}"
        )
        return

    contexts, aligned = align_logits_frames(context_to_path)

    (
        true_ids,
        soft_pred_ids,
        hard_pred_ids,
        label_names,
        per_context_preds,
    ) = compute_ensemble_predictions(contexts, aligned)


    group_dirname = f"{group_key}_C0toC3_ensemble"
    out_dir = os.path.join(ensemble_root, group_dirname)
    os.makedirs(out_dir, exist_ok=True)

    ref_ctx = contexts[0]
    ref_df = aligned[ref_ctx]


    out_df = pd.DataFrame()
    out_df["example_id"] = ref_df["example_id"]
    out_df["true_label_id"] = ref_df["true_label_id"]
    out_df["true_label"] = ref_df["true_label"]

    for ctx in contexts:
        preds = per_context_preds[ctx]
        out_df[f"{ctx}_pred_label_id"] = preds
        out_df[f"{ctx}_pred_label"] = [
            label_names[i] for i in preds
        ]

    out_df["ensemble_mean_logits_pred_label_id"] = soft_pred_ids
    out_df["ensemble_mean_logits_pred_label"] = [
        label_names[i] for i in soft_pred_ids
    ]
    out_df["ensemble_majority_vote_pred_label_id"] = hard_pred_ids
    out_df["ensemble_majority_vote_pred_label"] = [
        label_names[i] for i in hard_pred_ids
    ]


    logit_cols = [c for c in ref_df.columns if c.startswith("logit_")]
    logits_list = [
        aligned[ctx][logit_cols].to_numpy(dtype=float) for ctx in contexts
    ]
    stacked_logits = np.stack(logits_list, axis=0)
    mean_logits = stacked_logits.mean(axis=0)
    for idx, label_name in enumerate(label_names):
        out_df[f"ensemble_logit_{label_name}"] = mean_logits[:, idx]

    preds_csv_path = os.path.join(out_dir, "ensemble_predictions.csv")
    out_df.to_csv(preds_csv_path, index=False)


    soft_macro_f1, soft_acc, soft_report = compute_metrics(
        true_ids, soft_pred_ids, label_names
    )
    hard_macro_f1, hard_acc, hard_report = compute_metrics(
        true_ids, hard_pred_ids, label_names
    )

    report_lines: List[str] = []
    report_lines.append(
        f"Ensemble over contexts {', '.join(contexts)} for group: {group_key}"
    )
    for ctx in contexts:
        report_lines.append(f" - {ctx}: {context_to_path[ctx]}")
    report_lines.append("")

    report_lines.append("=== Ensemble (mean logits) on test set ===")
    report_lines.append(f"Macro F1 = {soft_macro_f1:.4f}")
    report_lines.append(f"Accuracy = {soft_acc:.4f}")
    report_lines.append("Classification report:")
    report_lines.append(soft_report)
    report_lines.append("")

    report_lines.append(
        "=== Ensemble (majority vote over context predictions) on test set ==="
    )
    report_lines.append(f"Macro F1 = {hard_macro_f1:.4f}")
    report_lines.append(f"Accuracy = {hard_acc:.4f}")
    report_lines.append("Classification report:")
    report_lines.append(hard_report)

    report_path = os.path.join(out_dir, "ensemble_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


    cm_soft_path = os.path.join(
        out_dir, "confusion_matrix_ensemble_mean_logits.png"
    )
    save_confusion_matrix(
        true_ids,
        soft_pred_ids,
        label_names,
        cm_soft_path,
        title="Confusion matrix (ensemble mean logits, test)",
    )

    cm_hard_path = os.path.join(
        out_dir, "confusion_matrix_ensemble_majority_vote.png"
    )
    save_confusion_matrix(
        true_ids,
        hard_pred_ids,
        label_names,
        cm_hard_path,
        title="Confusion matrix (ensemble majority vote, test)",
    )

    print(
        f"[OK] Group {group_key}: "
        f"mean-logits F1={soft_macro_f1:.4f}, acc={soft_acc:.4f}; "
        f"majority-vote F1={hard_macro_f1:.4f}, acc={hard_acc:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build C0–C3 context ensembles from saved test_logits.csv files "
            "in run_reports_advanced."
        )
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="run_reports_advanced",
        help="Directory containing per-run folders with test_logits.csv.",
    )
    parser.add_argument(
        "--ensemble_subdir",
        type=str,
        default="ensambled_results",
        help=(
            "Subfolder name inside report_dir where ensemble results "
            "will be written."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir
    ensemble_root = os.path.join(report_dir, args.ensemble_subdir)
    os.makedirs(ensemble_root, exist_ok=True)

    if not os.path.isdir(report_dir):
        raise SystemExit(f"Report directory not found: {report_dir}")

    groups = discover_groups(report_dir)
    if not groups:
        print(f"No groups with test_logits.csv found in {report_dir}.")
        return

    print(
        f"Discovered {len(groups)} base groups in {report_dir}. "
        "Processing those that have all C0–C3 contexts."
    )

    processed = 0
    for group_key, ctx_map in sorted(groups.items()):

        if all(c in ctx_map for c in CONTEXT_VALUES):
            process_group(group_key, ctx_map, ensemble_root)
            processed += 1
        else:
            missing = [c for c in CONTEXT_VALUES if c not in ctx_map]
            print(
                f"[SKIP] Group {group_key}: missing contexts "
                f"{', '.join(missing)}"
            )

    if processed == 0:
        print("No groups had all contexts C0–C3; nothing was ensembled.")
    else:
        print(
            f"Finished ensembling {processed} group(s). "
            f"Results are under: {ensemble_root}"
        )


if __name__ == "__main__":
    main()
