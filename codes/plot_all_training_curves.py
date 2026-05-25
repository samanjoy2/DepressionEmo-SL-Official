import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "matplotlib is required for plotting. "
        "Install it with: pip install matplotlib"
    ) from e

from collect_test_scores_advanced import parse_base_run_name


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


BACKBONE_ALIAS = {
    "FacebookAI-roberta-base": "RoBERTa-base",
    "google-bert-bert-base-uncased": "BERT-base",
    "mental-mental-bert-base-uncased": "MentalBERT-base",
    "mental-mental-roberta-base": "MentalRoBERTa-base",
    "microsoft-deberta-v3-base": "DeBERTaV3-base",
}

PREFERRED_MODEL_ORDER = [
    "RoBERTa-base",
    "BERT-base",
    "MentalBERT-base",
    "MentalRoBERTa-base",
    "DeBERTaV3-base",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-epoch training curves from all runs in "
            "run_reports_advanced and create a single comparison figure, "
            "with one row per backbone and lines for each context (C0–C3)."
        )
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="run_reports_advanced",
        help="Directory containing per-run folders with metrics_per_epoch.csv.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_macro_f1",
        choices=[
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "train_macro_f1",
            "val_macro_f1",
        ],
        help="Per-epoch metric to plot on the y-axis (default: val_macro_f1).",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="training_curves_all_models",
        help=(
            "Base filename (without extension) for the output figure. "
            "Files will be created as <out_prefix>_<metric>.png/.pdf "
            "inside report_dir."
        ),
    )
    return parser.parse_args()


def make_backbone_name(model_short: str) -> str:
    return BACKBONE_ALIAS.get(model_short, model_short)


def discover_curves(
    report_dir: str,
    metric: str,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    if not os.path.isdir(report_dir):
        raise SystemExit(f"Report directory not found: {report_dir}")

    curves: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    for entry in sorted(os.listdir(report_dir)):
        run_path = os.path.join(report_dir, entry)
        if not os.path.isdir(run_path):
            continue

        metrics_path = os.path.join(run_path, "metrics_per_epoch.csv")
        if not os.path.isfile(metrics_path):
            continue

        meta = parse_base_run_name(entry)
        model_short = meta.get("model_short")
        context = meta.get("context")

        if not model_short or not context:
            continue

        backbone = make_backbone_name(model_short)

        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"[WARN] Failed to read {metrics_path}: {e}")
            continue

        if metric not in df.columns:
            print(
                f"[WARN] Metric '{metric}' not found in {metrics_path}; "
                "skipping this run."
            )
            continue

        if "epoch" not in df.columns:
            print(
                f"[WARN] No 'epoch' column in {metrics_path}; "
                "skipping this run."
            )
            continue

        epochs = df["epoch"].to_numpy()
        values = df[metric].to_numpy()

        key = (backbone, context)


        if key in curves:
            continue
        curves[key] = (epochs, values)

    return curves


def plot_all_training_curves(
    curves: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    metric: str,
    out_dir: str,
    out_prefix: str,
) -> None:
    if not curves:
        print("[WARN] No curves found to plot.")
        return


    backbones_present: List[str] = sorted(
        {b for (b, _ctx) in curves.keys()},
        key=lambda b: (
            PREFERRED_MODEL_ORDER.index(b)
            if b in PREFERRED_MODEL_ORDER
            else len(PREFERRED_MODEL_ORDER)
        ),
    )

    n_models = len(backbones_present)
    context_colors = {
        "C0": "tab:blue",
        "C1": "tab:orange",
        "C2": "tab:green",
        "C3": "tab:red",
    }

    os.makedirs(out_dir, exist_ok=True)


    fig_height = max(5.0, 1.5 * n_models + 1.0)
    fig, axes = plt.subplots(
        n_models,
        1,
        sharex=True,
        figsize=(7.0, fig_height),
    )

    if n_models == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    for ax, backbone in zip(axes_list, backbones_present):
        for context, color in context_colors.items():
            key = (backbone, context)
            if key not in curves:
                continue
            epochs, values = curves[key]
            ax.plot(
                epochs,
                values,
                marker="o",
                linewidth=1.5,
                markersize=3,
                color=color,
                label=context,
            )

        ax.set_ylabel(backbone, rotation=0, labelpad=40, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes_list[-1].set_xlabel("Epoch")

    ylabel_map = {
        "train_loss": "Train loss",
        "val_loss": "Val loss",
        "train_accuracy": "Train accuracy",
        "val_accuracy": "Val accuracy",
        "train_macro_f1": "Train macro-F1",
        "val_macro_f1": "Val macro-F1",
    }
    y_label = ylabel_map.get(metric, metric)
    fig.supylabel(y_label, x=0.01)


    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=False,
            ncol=len(labels),
            fontsize=8,
        )

    fig.suptitle(f"{y_label} over epochs by backbone and context", fontsize=11)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])

    png_path = os.path.join(out_dir, f"{out_prefix}_{metric}.png")
    pdf_path = os.path.join(out_dir, f"{out_prefix}_{metric}.pdf")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved training-curves comparison: {png_path}")
    print(f"Saved training-curves comparison (PDF, selectable text): {pdf_path}")


def main() -> None:
    args = parse_args()
    curves = discover_curves(args.report_dir, args.metric)
    plot_all_training_curves(
        curves=curves,
        metric=args.metric,
        out_dir=args.report_dir,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
