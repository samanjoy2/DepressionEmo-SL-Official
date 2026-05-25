import argparse
import itertools
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ensemble_context_logits_advanced import (
    CONTEXT_VALUES,
    align_logits_frames,
    compute_ensemble_predictions,
    discover_groups,
    compute_metrics,
)
from collect_test_scores_advanced import parse_group_name_without_context


def generate_context_combinations(
    contexts: List[str],
    min_size: int = 2,
    max_size: int = None,
) -> List[Tuple[str, ...]]:
    if max_size is None:
        max_size = len(contexts)

    combos: List[Tuple[str, ...]] = []
    for r in range(min_size, max_size + 1):
        if r <= 0 or r > len(contexts):
            continue
        combos.extend(itertools.combinations(contexts, r))
    return combos


def compute_combination_metrics(
    contexts_subset: Tuple[str, ...],
    aligned: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, float]]:
    (
        true_ids,
        soft_pred_ids,
        hard_pred_ids,
        label_names,
        _,
    ) = compute_ensemble_predictions(list(contexts_subset), aligned)

    soft_macro_f1, soft_acc, _ = compute_metrics(
        true_ids, soft_pred_ids, label_names
    )
    hard_macro_f1, hard_acc, _ = compute_metrics(
        true_ids, hard_pred_ids, label_names
    )

    return {
        "mean_logits": {
            "macro_f1": float(soft_macro_f1),
            "accuracy": float(soft_acc),
        },
        "majority_vote": {
            "macro_f1": float(hard_macro_f1),
            "accuracy": float(hard_acc),
        },
    }


def build_combinations_summary(
    report_dir: str,
    min_size: int = 2,
    max_size: int = 4,
) -> pd.DataFrame:
    if not os.path.isdir(report_dir):
        raise SystemExit(f"Report directory not found: {report_dir}")

    groups = discover_groups(report_dir)
    if not groups:
        raise SystemExit(
            f"No groups with test_logits.csv found in {report_dir}."
        )

    rows: List[Dict[str, object]] = []

    for group_key, ctx_map in sorted(groups.items()):

        if not all(c in ctx_map for c in CONTEXT_VALUES):
            continue

        try:
            contexts, aligned = align_logits_frames(ctx_map)
        except Exception as e:
            print(
                f"[WARN] Skipping group {group_key} due to alignment error: {e}"
            )
            continue

        combos = generate_context_combinations(
            contexts, min_size=min_size, max_size=max_size
        )
        if not combos:
            continue

        meta = parse_group_name_without_context(group_key)

        for combo in combos:
            metrics_by_method = compute_combination_metrics(combo, aligned)
            combo_str = "+".join(combo)

            for method_name, metric_values in metrics_by_method.items():
                rows.append(
                    {
                        "group_key": group_key,
                        "model_short": meta.get("model_short"),
                        "epochs": meta.get("epochs"),
                        "seed": meta.get("seed"),
                        "freeze_layers": meta.get("freeze_layers"),
                        "contexts": combo_str,
                        "num_contexts": len(combo),
                        "ensemble_method": method_name,
                        "macro_f1": metric_values.get("macro_f1"),
                        "accuracy": metric_values.get("accuracy"),
                    }
                )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ensemble metrics for all multi-context "
            "combinations (e.g., pairs, triples, all four) "
            "for runs in run_reports_advanced that have C0-C3."
        )
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="run_reports_advanced",
        help=(
            "Directory containing per-run folders with test_logits.csv "
            "(same as for ensemble_context_logits_advanced.py)."
        ),
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="context_ensemble_combinations_summary.csv",
        help=(
            "Name of the output CSV file (created inside report_dir) "
            "with one row per (group, combination, ensemble_method)."
        ),
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=2,
        help="Minimum number of contexts in a combination (default: 2).",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=4,
        help=(
            "Maximum number of contexts in a combination "
            "(default: 4, i.e. up to C0+C1+C2+C3)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir

    if args.min_size <= 0:
        raise SystemExit("--min_size must be >= 1")
    if args.max_size < args.min_size:
        raise SystemExit("--max_size must be >= --min_size")

    df = build_combinations_summary(
        report_dir=report_dir,
        min_size=args.min_size,
        max_size=args.max_size,
    )

    if df.empty:
        print("No valid context combinations found; nothing to write.")
        return

    out_path = os.path.join(report_dir, args.out_csv)
    df.to_csv(out_path, index=False)
    print(
        f"Wrote {len(df)} rows with context ensemble combinations to {out_path}"
    )


if __name__ == "__main__":
    main()
