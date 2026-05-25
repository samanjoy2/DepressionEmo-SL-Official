import os
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from collect_test_scores_advanced import parse_base_run_name, parse_group_name_without_context
from ensemble_context_logits_advanced import CONTEXT_VALUES, discover_groups


def compute_label_mapping(
    df: pd.DataFrame,
    true_label_id_col: str = "true_label_id",
    true_label_col: str = "true_label",
) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for label_id, label_name in zip(df[true_label_id_col], df[true_label_col]):
        if int(label_id) not in mapping:
            mapping[int(label_id)] = str(label_name)
    return mapping


def compute_per_label_rows_from_df(
    df: pd.DataFrame,
    run_name: str,
    run_type: str,
    ensemble_method: str,
    model_short: Optional[str],
    context: Optional[str],
    epochs: Optional[int],
    seed: Optional[int],
    freeze_layers: Optional[int],
    report_dir: str,
    source_path: str,
    pred_label_id_col: str,
    true_label_id_col: str = "true_label_id",
    true_label_col: str = "true_label",
) -> List[Dict[str, object]]:
    if true_label_id_col not in df.columns or pred_label_id_col not in df.columns:
        return []
    if true_label_col not in df.columns:
        return []

    y_true = df[true_label_id_col].to_numpy()
    y_pred = df[pred_label_id_col].to_numpy()


    label_ids = sorted(int(x) for x in np.unique(y_true))
    if not label_ids:
        return []

    id_to_name = compute_label_mapping(
        df, true_label_id_col=true_label_id_col, true_label_col=true_label_col
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )

    rel_source = os.path.relpath(source_path, report_dir)

    rows: List[Dict[str, object]] = []
    for idx, label_id in enumerate(label_ids):
        rows.append(
            {
                "run_name": run_name,
                "run_type": run_type,
                "ensemble_method": ensemble_method,
                "model_short": model_short,
                "context": context,
                "epochs": epochs,
                "seed": seed,
                "freeze_layers": freeze_layers,
                "label_id": int(label_id),
                "label": id_to_name.get(int(label_id)),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "source_file": rel_source,
            }
        )

    return rows


def collect_per_label_scores(
    report_dir: str,
    ensemble_subdir: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []


    groups = discover_groups(report_dir)
    if not groups:
        return rows


    for group_key, context_to_path in sorted(groups.items()):
        if not all(c in context_to_path for c in CONTEXT_VALUES):
            continue

        for context in CONTEXT_VALUES:
            run_path = context_to_path[context]
            logits_path = os.path.join(run_path, "test_logits.csv")
            if not os.path.isfile(logits_path):
                continue

            run_name = os.path.basename(run_path)
            meta = parse_base_run_name(run_name)

            df_logits = pd.read_csv(logits_path)

            rows.extend(
                compute_per_label_rows_from_df(
                    df=df_logits,
                    run_name=run_name,
                    run_type="base",
                    ensemble_method="",
                    model_short=meta.get("model_short"),
                    context=meta.get("context"),
                    epochs=meta.get("epochs"),
                    seed=meta.get("seed"),
                    freeze_layers=meta.get("freeze_layers"),
                    report_dir=report_dir,
                    source_path=logits_path,
                    pred_label_id_col="pred_label_id",
                )
            )


    ensemble_root = os.path.join(report_dir, ensemble_subdir)
    if os.path.isdir(ensemble_root):
        for entry in sorted(os.listdir(ensemble_root)):
            full_path = os.path.join(ensemble_root, entry)
            if not os.path.isdir(full_path):
                continue

            preds_path = os.path.join(full_path, "ensemble_predictions.csv")
            if not os.path.isfile(preds_path):
                continue


            parts = entry.split("_")
            if len(parts) < 4 or parts[-2:] != ["C0toC3", "ensemble"]:
                group_name = entry
                context = "ensemble"
            else:
                group_name = "_".join(parts[:-2])
                context = "C0toC3"

            meta = parse_group_name_without_context(group_name)

            df_preds = pd.read_csv(preds_path)

            method_to_column = {
                "mean_logits": "ensemble_mean_logits_pred_label_id",
                "majority_vote": "ensemble_majority_vote_pred_label_id",
            }

            for method_name, col in method_to_column.items():
                if col not in df_preds.columns:
                    continue

                rows.extend(
                    compute_per_label_rows_from_df(
                        df=df_preds,
                        run_name=entry,
                        run_type="ensemble",
                        ensemble_method=method_name,
                        model_short=meta.get("model_short"),
                        context=context,
                        epochs=meta.get("epochs"),
                        seed=meta.get("seed"),
                        freeze_layers=meta.get("freeze_layers"),
                        report_dir=report_dir,
                        source_path=preds_path,
                        pred_label_id_col=col,
                    )
                )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-label precision/recall/F1/support from "
            "run_reports_advanced test predictions (C0–C3 groups only) "
            "and write them into a CSV."
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
        help="Subdirectory inside report_dir containing ensemble runs.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="label_scores_summary.csv",
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

    rows = collect_per_label_scores(report_dir, ensemble_subdir)
    if not rows:
        print(f"No per-label scores found under {report_dir}.")
        return

    df = pd.DataFrame(rows)
    out_path = os.path.join(report_dir, out_csv_name)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} per-label rows to {out_path}")


if __name__ == "__main__":
    main()
