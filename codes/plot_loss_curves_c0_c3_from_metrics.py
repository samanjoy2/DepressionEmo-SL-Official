import os
import argparse
from typing import Dict, Tuple, List

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


MODEL_ALIAS: Dict[str, str] = {
    "FacebookAI-roberta-base": "RoBERTa-base",
    "google-bert-bert-base-uncased": "BERT-base",
    "mental-mental-bert-base-uncased": "MentalBERT-base",
    "mental-mental-roberta-base": "MentalRoBERTa-base",
    "microsoft-deberta-v3-base": "DeBERTaV3-base",
}

PREFERRED_ORDER: List[str] = [
    "RoBERTa-base",
    "BERT-base",
    "MentalBERT-base",
    "MentalRoBERTa-base",
    "DeBERTaV3-base",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild train/validation loss curves for all contexts (C0–C3) "
            "from metrics_per_epoch.csv, and plot them in a single figure "
            "with one row per backbone and one column per context."
        )
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="run_reports_advanced",
        help="Directory containing per-run folders with metrics_per_epoch.csv.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="all_models_train_val_loss_C0toC3",
        help="Base filename (without extension) for the output figure.",
    )
    return parser.parse_args()


def discover_loss_curves(
    report_dir: str,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not os.path.isdir(report_dir):
        raise SystemExit(f"Report directory not found: {report_dir}")

    curves: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for entry in os.listdir(report_dir):
        run_path = os.path.join(report_dir, entry)
        if not os.path.isdir(run_path):
            continue

        meta = parse_base_run_name(entry)
        model_short = meta.get("model_short")
        context = meta.get("context")

        if not model_short or context not in {"C0", "C1", "C2", "C3"}:
            continue

        backbone = MODEL_ALIAS.get(model_short, model_short)

        metrics_path = os.path.join(run_path, "metrics_per_epoch.csv")
        if not os.path.isfile(metrics_path):
            continue

        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue

        required_cols = {"epoch", "train_loss", "val_loss"}
        if not required_cols.issubset(df.columns):
            continue

        epochs = df["epoch"].to_numpy()
        train_loss = df["train_loss"].to_numpy()
        val_loss = df["val_loss"].to_numpy()

        key = (backbone, context)

        if key not in curves:
            curves[key] = (epochs, train_loss, val_loss)

    return curves


def plot_all_models_loss(
    curves: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    report_dir: str,
    out_name: str,
) -> None:
    if not curves:
        print("[WARN] No loss curves found for C0–C3.")
        return

    backbones_present = sorted(
        {b for (b, _c) in curves.keys()},
        key=lambda b: PREFERRED_ORDER.index(b)
        if b in PREFERRED_ORDER
        else len(PREFERRED_ORDER),
    )

    n_rows = len(backbones_present)
    if n_rows == 0:
        print("[WARN] No backbones with C0–C3 curves available.")
        return


    contexts: List[str] = ["C0", "C1", "C2", "C3"]
    n_cols = len(contexts)

    fig_height = max(4.0, 2.0 * n_rows + 1.0)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.0, fig_height),
        sharex=True,
        sharey=True,
    )


    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, backbone in enumerate(backbones_present):
        for col_idx, context in enumerate(contexts):
            ax = axes[row_idx, col_idx]
            key = (backbone, context)
            if key not in curves:
                ax.set_visible(False)
                continue

            epochs, train_loss, val_loss = curves[key]
            ax.plot(
                epochs,
                train_loss,
                label="Train loss",
                color="tab:blue",
                linewidth=1.5,
            )
            ax.plot(
                epochs,
                val_loss,
                label="Validation loss",
                color="tab:orange",
                linewidth=1.5,
            )

            if row_idx == 0:
                ax.set_title(context)

            ax.grid(True, linestyle="--", alpha=0.3)


    for col_idx in range(n_cols):
        axes[-1, col_idx].set_xlabel("Epoch")


    fig.supylabel("Loss", x=0.07)


    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=False,
            fontsize=8,
        )

    fig.suptitle(
        "Train vs validation loss across contexts (C0–C3)",
        fontsize=11,
        y=0.965,
    )


    fig.tight_layout(rect=[0.12, 0.06, 0.98, 0.93])


    fig.canvas.draw()
    for row_idx, backbone in enumerate(backbones_present):
        row_axes = axes[row_idx, :]

        ref_ax = next((ax for ax in row_axes if ax.get_visible()), row_axes[0])
        bbox = ref_ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        fig.text(
            0.03,
            y_center,
            backbone,
            va="center",
            ha="center",
            rotation=90,
            fontsize=8,
        )

    os.makedirs(report_dir, exist_ok=True)
    png_path = os.path.join(report_dir, f"{out_name}.png")
    pdf_path = os.path.join(report_dir, f"{out_name}.pdf")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined loss figure: {png_path}")
    print(f"Saved combined loss figure (PDF, selectable text): {pdf_path}")


def main() -> None:
    args = parse_args()
    curves = discover_loss_curves(args.report_dir)
    plot_all_models_loss(curves, args.report_dir, args.out_name)


if __name__ == "__main__":
    main()
