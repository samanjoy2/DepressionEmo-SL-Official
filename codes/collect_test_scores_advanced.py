import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def parse_base_run_name(name: str) -> Dict[str, Optional[object]]:
    parts = name.split("_")
    info: Dict[str, Optional[object]] = {
        "model_short": None,
        "context": None,
        "epochs": None,
        "seed": None,
        "freeze_layers": None,
    }

    if len(parts) < 5:
        return info

    context = parts[-4]
    epochs_part = parts[-3]
    seed_part = parts[-2]
    freeze_part = parts[-1]
    model_short = "_".join(parts[:-4])

    info["model_short"] = model_short
    info["context"] = context

    m_epochs = re.match(r"(\d+)epochs$", epochs_part)
    m_seed = re.match(r"seed(\d+)$", seed_part)
    m_freeze = re.match(r"freeze(\d+)$", freeze_part)

    if m_epochs:
        info["epochs"] = int(m_epochs.group(1))
    if m_seed:
        info["seed"] = int(m_seed.group(1))
    if m_freeze:
        info["freeze_layers"] = int(m_freeze.group(1))

    return info


def parse_group_name_without_context(name: str) -> Dict[str, Optional[object]]:
    parts = name.split("_")
    info: Dict[str, Optional[object]] = {
        "model_short": None,
        "epochs": None,
        "seed": None,
        "freeze_layers": None,
    }

    if len(parts) < 4:
        return info

    epochs_part = parts[-3]
    seed_part = parts[-2]
    freeze_part = parts[-1]
    model_short = "_".join(parts[:-3])

    info["model_short"] = model_short

    m_epochs = re.match(r"(\d+)epochs$", epochs_part)
    m_seed = re.match(r"seed(\d+)$", seed_part)
    m_freeze = re.match(r"freeze(\d+)$", freeze_part)

    if m_epochs:
        info["epochs"] = int(m_epochs.group(1))
    if m_seed:
        info["seed"] = int(m_seed.group(1))
    if m_freeze:
        info["freeze_layers"] = int(m_freeze.group(1))

    return info


def extract_base_test_metrics(run_report_path: str) -> Optional[Tuple[float, float]]:
    if not os.path.isfile(run_report_path):
        return None

    macro_f1: Optional[float] = None
    accuracy: Optional[float] = None

    with open(run_report_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("Test macro F1"):
                try:
                    macro_f1 = float(s.split("=", 1)[1].strip())
                except Exception:
                    pass
            elif s.startswith("Test accuracy"):
                try:
                    accuracy = float(s.split("=", 1)[1].strip())
                except Exception:
                    pass

    if macro_f1 is None or accuracy is None:
        return None
    return macro_f1, accuracy


def compute_metrics_from_ids(
    y_true, y_pred
) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)

    metrics: Dict[str, float] = {}


    acc = accuracy_score(y_true, y_pred)
    metrics["accuracy"] = float(acc)


    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    metrics["macro_f1"] = float(f_macro)
    metrics["precision_macro"] = float(p_macro)
    metrics["recall_macro"] = float(r_macro)

    metrics["f1_weighted"] = float(f_weighted)
    metrics["precision_weighted"] = float(p_weighted)
    metrics["recall_weighted"] = float(r_weighted)

    metrics["f1_micro"] = float(f_micro)
    metrics["precision_micro"] = float(p_micro)
    metrics["recall_micro"] = float(r_micro)


    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y_true, y_pred)
    )
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))

    return metrics


def compute_metrics_from_logits_csv(
    logits_path: str,
) -> Optional[Dict[str, float]]:
    if not os.path.isfile(logits_path):
        return None

    df = pd.read_csv(logits_path)
    if "true_label_id" not in df.columns or "pred_label_id" not in df.columns:
        return None

    y_true = df["true_label_id"].to_numpy()
    y_pred = df["pred_label_id"].to_numpy()
    return compute_metrics_from_ids(y_true, y_pred)


def extract_ensemble_metrics(
    ensemble_report_path: str,
) -> Dict[str, Tuple[float, float]]:
    results: Dict[str, Tuple[float, float]] = {}

    if not os.path.isfile(ensemble_report_path):
        return results

    current_method: Optional[str] = None
    macro_f1: Optional[float] = None
    accuracy: Optional[float] = None

    with open(ensemble_report_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("=== Ensemble (mean logits)"):

                if current_method and macro_f1 is not None and accuracy is not None:
                    results[current_method] = (macro_f1, accuracy)
                current_method = "mean_logits"
                macro_f1 = None
                accuracy = None
                continue
            if s.startswith("=== Ensemble (majority vote over context predictions)"):
                if current_method and macro_f1 is not None and accuracy is not None:
                    results[current_method] = (macro_f1, accuracy)
                current_method = "majority_vote"
                macro_f1 = None
                accuracy = None
                continue

            if current_method is None:
                continue

            if s.startswith("Macro F1"):
                try:
                    macro_f1 = float(s.split("=", 1)[1].strip())
                except Exception:
                    pass
            elif s.startswith("Accuracy"):
                try:
                    accuracy = float(s.split("=", 1)[1].strip())
                except Exception:
                    pass

    if current_method and macro_f1 is not None and accuracy is not None:
        results[current_method] = (macro_f1, accuracy)

    return results


def collect_scores(
    report_dir: str,
    ensemble_subdir: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []


    for entry in os.listdir(report_dir):
        full_path = os.path.join(report_dir, entry)
        if not os.path.isdir(full_path):
            continue

        if entry == ensemble_subdir:
            continue

        if entry.startswith("[") and entry.endswith("]"):
            continue

        run_report_path = os.path.join(full_path, "run_report.txt")
        if not os.path.isfile(run_report_path):
            continue


        logits_path = os.path.join(full_path, "test_logits.csv")
        metrics = compute_metrics_from_logits_csv(logits_path)


        if metrics is None:
            base_metrics = extract_base_test_metrics(run_report_path)
            if base_metrics is None:
                continue
            macro_f1, accuracy = base_metrics
            metrics = {
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "precision_macro": None,
                "recall_macro": None,
                "f1_weighted": None,
                "precision_weighted": None,
                "recall_weighted": None,
                "f1_micro": None,
                "precision_micro": None,
                "recall_micro": None,
                "balanced_accuracy": None,
                "cohen_kappa": None,
                "matthews_corrcoef": None,
            }

        meta = parse_base_run_name(entry)

        rows.append(
            {
                "run_name": entry,
                "run_type": "base",
                "ensemble_method": "",
                "model_short": meta.get("model_short"),
                "context": meta.get("context"),
                "epochs": meta.get("epochs"),
                "seed": meta.get("seed"),
                "freeze_layers": meta.get("freeze_layers"),
                "macro_f1": metrics.get("macro_f1"),
                "accuracy": metrics.get("accuracy"),
                "precision_macro": metrics.get("precision_macro"),
                "recall_macro": metrics.get("recall_macro"),
                "f1_weighted": metrics.get("f1_weighted"),
                "precision_weighted": metrics.get("precision_weighted"),
                "recall_weighted": metrics.get("recall_weighted"),
                "f1_micro": metrics.get("f1_micro"),
                "precision_micro": metrics.get("precision_micro"),
                "recall_micro": metrics.get("recall_micro"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "cohen_kappa": metrics.get("cohen_kappa"),
                "matthews_corrcoef": metrics.get("matthews_corrcoef"),
                "source_file": os.path.relpath(run_report_path, report_dir),
            }
        )


    ensemble_root = os.path.join(report_dir, ensemble_subdir)
    if os.path.isdir(ensemble_root):
        for entry in os.listdir(ensemble_root):
            full_path = os.path.join(ensemble_root, entry)
            if not os.path.isdir(full_path):
                continue

            ensemble_report_path = os.path.join(full_path, "ensemble_report.txt")


            metrics_by_method: Dict[str, Dict[str, float]] = {}
            preds_path = os.path.join(full_path, "ensemble_predictions.csv")
            if os.path.isfile(preds_path):
                df_preds = pd.read_csv(preds_path)
                if "true_label_id" in df_preds.columns:
                    y_true = df_preds["true_label_id"].to_numpy()
                    method_to_column = {
                        "mean_logits": "ensemble_mean_logits_pred_label_id",
                        "majority_vote": "ensemble_majority_vote_pred_label_id",
                    }
                    for method_name, col in method_to_column.items():
                        if col in df_preds.columns:
                            y_pred = df_preds[col].to_numpy()
                            metrics_by_method[method_name] = compute_metrics_from_ids(
                                y_true, y_pred
                            )


            basic_metrics = extract_ensemble_metrics(ensemble_report_path)
            for method_name, (macro_f1, accuracy) in basic_metrics.items():
                if method_name not in metrics_by_method:
                    metrics_by_method[method_name] = {
                        "macro_f1": macro_f1,
                        "accuracy": accuracy,
                        "precision_macro": None,
                        "recall_macro": None,
                        "f1_weighted": None,
                        "precision_weighted": None,
                        "recall_weighted": None,
                        "f1_micro": None,
                        "precision_micro": None,
                        "recall_micro": None,
                        "balanced_accuracy": None,
                        "cohen_kappa": None,
                        "matthews_corrcoef": None,
                    }

            if not metrics_by_method:
                continue


            parts = entry.split("_")
            if len(parts) < 4 or parts[-2:] != ["C0toC3", "ensemble"]:
                group_name = entry
                context = "ensemble"
            else:
                group_name = "_".join(parts[:-2])
                context = "C0toC3"

            meta = parse_group_name_without_context(group_name)

            for method_name, metrics in metrics_by_method.items():
                rows.append(
                    {
                        "run_name": entry,
                        "run_type": "ensemble",
                        "ensemble_method": method_name,
                        "model_short": meta.get("model_short"),
                        "context": context,
                        "epochs": meta.get("epochs"),
                        "seed": meta.get("seed"),
                        "freeze_layers": meta.get("freeze_layers"),
                        "macro_f1": metrics.get("macro_f1"),
                        "accuracy": metrics.get("accuracy"),
                        "precision_macro": metrics.get("precision_macro"),
                        "recall_macro": metrics.get("recall_macro"),
                        "f1_weighted": metrics.get("f1_weighted"),
                        "precision_weighted": metrics.get("precision_weighted"),
                        "recall_weighted": metrics.get("recall_weighted"),
                        "f1_micro": metrics.get("f1_micro"),
                        "precision_micro": metrics.get("precision_micro"),
                        "recall_micro": metrics.get("recall_micro"),
                        "balanced_accuracy": metrics.get("balanced_accuracy"),
                        "cohen_kappa": metrics.get("cohen_kappa"),
                        "matthews_corrcoef": metrics.get("matthews_corrcoef"),
                        "source_file": os.path.relpath(
                            ensemble_report_path, report_dir
                        ),
                    }
                )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect final test macro F1 and accuracy from run_reports_advanced "
            "and write them into a CSV."
        )
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="run_reports_advanced",
        help="Directory containing per-run folders (with run_report.txt).",
    )
    parser.add_argument(
        "--ensemble_subdir",
        type=str,
        default="ensambled_results",
        help="Subdirectory inside report_dir containing ensemble runs.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="test_scores_summary.csv",
        help="Name of the output CSV file (created inside report_dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir
    ensemble_subdir = args.ensemble_subdir
    out_csv_name = args.out_csv

    if not os.path.isdir(report_dir):
        raise SystemExit(f"Report directory not found: {report_dir}")

    rows = collect_scores(report_dir, ensemble_subdir)
    if not rows:
        print(f"No test scores found under {report_dir}.")
        return

    df = pd.DataFrame(rows)
    out_path = os.path.join(report_dir, out_csv_name)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
