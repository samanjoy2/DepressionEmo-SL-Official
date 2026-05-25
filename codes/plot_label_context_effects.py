import os
import argparse
from typing import List, Tuple

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "matplotlib is required for plotting. "
        "Install it with: pip install matplotlib"
    ) from e


CONTEXT_ORDER: List[str] = ["C0", "C1", "C2", "C3", "C0toC3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use label_scores_summary.csv to plot, per model/config, how "
            "per-label scores change with more context (C0–C3) and which "
            "labels benefit most."
        )
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=os.path.join("run_reports_advanced", "label_scores_summary.csv"),
        help="Path to label_scores_summary.csv.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("run_reports_advanced", "label_context_plots"),
        help="Output directory where plots will be saved.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Which per-label metric to plot (default: f1).",
    )
    return parser.parse_args()


def prepare_base_dataframe(summary_csv: str) -> pd.DataFrame:
    if not os.path.isfile(summary_csv):
        raise SystemExit(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)


    df = df[df["run_type"].isin(["base", "ensemble"])].copy()
    df = df[df["context"].isin(CONTEXT_ORDER)].copy()

    if df.empty:
        raise SystemExit("No base runs with contexts C0–C3 found in the CSV.")

    return df


def make_group_name(
    model_short: str,
    epochs: float,
    seed: float,
    freeze_layers: float,
) -> str:
    return (
        f"{model_short}_epochs{int(epochs)}_seed{int(seed)}_freeze{int(freeze_layers)}"
    )


def plot_metric_by_context_for_group(
    df: pd.DataFrame,
    metric: str,
    out_dir: str,
    group_name: str,
) -> None:
    agg = df.groupby(["label", "context"])[metric].mean().reset_index()

    pivot = agg.pivot(index="context", columns="label", values=metric)
    pivot = pivot.reindex(index=[c for c in CONTEXT_ORDER if c in pivot.index])

    if pivot.empty:
        print("[WARN] No data available for metric-by-context plot (group).")
        return

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    contexts = list(pivot.index)
    x_positions = list(range(len(contexts)))

    for label in pivot.columns:
        y_values = pivot[label].values
        valid_mask = ~pd.isna(y_values)

        if valid_mask.sum() < 2:
            continue

        x_valid = np.array(x_positions)[valid_mask]
        y_valid = y_values[valid_mask]


        degree = min(4, len(x_valid) - 1)

        try:
            coeffs = np.polyfit(x_valid, y_valid, degree)
            x_smooth = np.linspace(x_valid.min(), x_valid.max(), 200)
            y_smooth = np.polyval(coeffs, x_smooth)

            (line,) = plt.plot(
                x_smooth,
                y_smooth,
                label=label,
            )
            plt.scatter(
                x_valid,
                y_valid,
                color=line.get_color(),
                s=25,
            )
        except np.linalg.LinAlgError:
            plt.plot(
                x_valid,
                y_valid,
                marker="o",
                label=label,
            )

    plt.xticks(x_positions, contexts)
    plt.xlabel("Context")
    plt.ylabel(metric.upper())
    plt.title(f"Per-label {metric.upper()} vs. context\nModel: {group_name}")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"context_vs_label_{metric}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_delta_C3_minus_C0_for_group(
    df: pd.DataFrame,
    metric: str,
    out_dir: str,
    group_name: str,
) -> None:
    agg = df.groupby(["label", "context"])[metric].mean().reset_index()
    pivot = agg.pivot(index="label", columns="context", values=metric)

    if "C0" not in pivot.columns or "C3" not in pivot.columns:
        print(
            "[WARN] Cannot compute C3-C0 deltas for this model: "
            "missing C0 or C3 in data."
        )
        return

    delta = pivot["C3"] - pivot["C0"]
    delta = delta.sort_values(ascending=False)

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))

    labels = delta.index.tolist()
    values = delta.values

    plt.bar(range(len(labels)), values, color="steelblue")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(f"{metric.upper()}(C3) - {metric.upper()}(C0)")
    plt.title(
        f"Change in per-label {metric.upper()} from C0 to C3\nModel: {group_name}"
    )
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"delta_{metric}_C3_minus_C0_by_label.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    args = parse_args()

    df = prepare_base_dataframe(args.summary_csv)

    if args.metric not in df.columns:
        raise SystemExit(f"Metric column '{args.metric}' not found in CSV.")


    group_cols: List[str] = ["model_short", "epochs", "seed", "freeze_layers"]

    if not all(col in df.columns for col in group_cols):
        missing = [c for c in group_cols if c not in df.columns]
        raise SystemExit(f"Missing required columns in CSV: {', '.join(missing)}")

    for (model_short, epochs, seed, freeze_layers), df_group in df.groupby(group_cols):
        group_name = make_group_name(model_short, epochs, seed, freeze_layers)
        group_dir = os.path.join(args.out_dir, group_name)
        print(f"Processing model group: {group_name}")
        plot_metric_by_context_for_group(df_group, args.metric, group_dir, group_name)
        plot_delta_C3_minus_C0_for_group(df_group, args.metric, group_dir, group_name)


if __name__ == "__main__":
    main()
