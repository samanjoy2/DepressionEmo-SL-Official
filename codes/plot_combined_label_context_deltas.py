import os
import argparse
from typing import List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "matplotlib is required for plotting. "
        "Install it with: pip install matplotlib"
    ) from e


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build combined plots showing, for each model and label, "
            "how much the chosen metric changes from C0 to C3, "
            "and which labels benefit most from added context."
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
        default=os.path.join("run_reports_advanced", "combined_label_context_plots"),
        help="Output directory where plots will be saved.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Per-label metric to use for C3-C0 deltas (default: f1).",
    )
    parser.add_argument(
        "--top_k_labels",
        type=int,
        default=0,
        help=(
            "If > 0, bar plot will show only the top-K labels by "
            "average C3-C0 delta across models (default: 0 = all labels)."
        ),
    )
    return parser.parse_args()


def make_group_name(
    model_short: str,
    epochs: float,
    seed: float,
    freeze_layers: float,
) -> str:
    alias_map = {
        "FacebookAI-roberta-base": "RoBERTa-base",
        "google-bert-bert-base-uncased": "BERT-base",
        "mental-mental-bert-base-uncased": "MentalBERT-base",
        "mental-mental-roberta-base": "MentalRoBERTa-base",
        "microsoft-deberta-v3-base": "DeBERTaV3-base",
    }

    return alias_map.get(model_short, model_short)


def load_c0_c3_deltas(
    summary_csv: str,
    metric: str,
) -> pd.DataFrame:
    if not os.path.isfile(summary_csv):
        raise SystemExit(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)

    required_cols: List[str] = [
        "run_type",
        "model_short",
        "context",
        "epochs",
        "seed",
        "freeze_layers",
        "label_id",
        "label",
        metric,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing required columns in summary CSV: {', '.join(missing)}"
        )


    df = df[df["run_type"] == "base"].copy()
    df = df[df["context"].isin(["C0", "C3"])].copy()

    if df.empty:
        raise SystemExit("No base runs with contexts C0 and C3 found in the CSV.")


    group_cols = [
        "model_short",
        "epochs",
        "seed",
        "freeze_layers",
        "label_id",
        "label",
        "context",
    ]
    agg = df.groupby(group_cols)[metric].mean().reset_index()


    pivot = agg.pivot_table(
        index=[
            "model_short",
            "epochs",
            "seed",
            "freeze_layers",
            "label_id",
            "label",
        ],
        columns="context",
        values=metric,
    )


    if "C0" not in pivot.columns or "C3" not in pivot.columns:
        raise SystemExit(
            "Cannot compute C3-C0 deltas: missing C0 or C3 in aggregated data."
        )

    pivot = pivot.dropna(subset=["C0", "C3"])
    if pivot.empty:
        raise SystemExit("No labels with both C0 and C3 measurements.")

    pivot["delta"] = pivot["C3"] - pivot["C0"]
    pivot = pivot.reset_index()


    pivot["model_name"] = pivot.apply(
        lambda r: make_group_name(
            r["model_short"], r["epochs"], r["seed"], r["freeze_layers"]
        ),
        axis=1,
    )

    return pivot[
        [
            "model_name",
            "model_short",
            "epochs",
            "seed",
            "freeze_layers",
            "label_id",
            "label",
            "delta",
        ]
    ]


def plot_heatmap(
    df_delta: pd.DataFrame,
    metric: str,
    out_dir: str,
) -> None:
    pivot = df_delta.pivot_table(
        index="model_name", columns="label", values="delta", aggfunc="mean"
    )

    preferred_models = [
        "RoBERTa-base",
        "BERT-base",
        "MentalBERT-base",
        "MentalRoBERTa-base",
        "DeBERTaV3-base",
    ]
    row_order = [
        m for m in preferred_models if m in pivot.index
    ] + [m for m in pivot.index if m not in preferred_models]
    pivot = pivot.loc[row_order]
    pivot = pivot[pivot.columns.sort_values()]

    if pivot.empty:
        print("[WARN] No data for heatmap plot.")
        return

    os.makedirs(out_dir, exist_ok=True)


    n_labels = pivot.shape[1]
    n_models = pivot.shape[0]
    width = min(7.0, max(5.0, 0.35 * n_labels + 2.0))
    height = max(4.5, 0.8 * n_models + 1.5)

    plt.figure(figsize=(width, height))
    im = plt.imshow(pivot.values, aspect="auto", cmap="bwr", vmin=-0.3, vmax=0.3)
    plt.colorbar(im, label=f"{metric.upper()}(C3) - {metric.upper()}(C0)")

    plt.xticks(
        ticks=np.arange(pivot.shape[1]),
        labels=pivot.columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        ticks=np.arange(pivot.shape[0]),
        labels=pivot.index,
    )


    for x_pos in np.arange(-0.5, n_labels, 1.0):
        plt.axvline(x_pos + 0.5, color="white", linewidth=0.4, alpha=0.7)
    for y_pos in np.arange(-0.5, n_models, 1.0):
        plt.axhline(y_pos + 0.5, color="white", linewidth=0.4, alpha=0.7)

    plt.xlabel("Label")
    plt.ylabel("Backbone")
    plt.title(f"C3-C0 {metric.upper()} delta per label and model")
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"heatmap_delta_{metric}_C3_minus_C0.png")
    pdf_path = os.path.join(out_dir, f"heatmap_delta_{metric}_C3_minus_C0.pdf")
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap: {png_path}")
    print(f"Saved heatmap (PDF, selectable text): {pdf_path}")


def plot_top_labels_bar(
    df_delta: pd.DataFrame,
    metric: str,
    out_dir: str,
    top_k_labels: int = 0,
) -> None:

    label_mean = (
        df_delta.groupby("label")["delta"].mean().sort_values(ascending=False)
    )
    if top_k_labels > 0:
        label_order = label_mean.index[:top_k_labels].tolist()
    else:
        label_order = label_mean.index.tolist()

    if not label_order:
        print("[WARN] No labels available for bar plot.")
        return


    pivot = df_delta.pivot_table(
        index="label", columns="model_name", values="delta", aggfunc="mean"
    )
    pivot = pivot.loc[[l for l in label_order if l in pivot.index]]


    preferred_models = [
        "RoBERTa-base",
        "BERT-base",
        "MentalBERT-base",
        "MentalRoBERTa-base",
        "DeBERTaV3-base",
    ]
    col_order = [
        m for m in preferred_models if m in pivot.columns
    ] + [m for m in pivot.columns if m not in preferred_models]
    pivot = pivot[col_order]

    if pivot.empty:
        print("[WARN] No data for bar plot.")
        return

    models = pivot.columns.tolist()
    num_labels = pivot.shape[0]
    num_models = len(models)

    x = np.arange(num_labels)
    width = 0.8 / max(num_models, 1)

    os.makedirs(out_dir, exist_ok=True)


    plt.figure(figsize=(7.0, 5.5))

    for idx, model_name in enumerate(models):
        y = pivot[model_name].values
        bars = plt.bar(
            x + (idx - num_models / 2) * width + width / 2,
            y,
            width=width,
            label=model_name,
        )


        for rect in bars:
            height = rect.get_height()
            if np.isnan(height):
                continue

            x_pos = rect.get_x() + rect.get_width() / 2.0
            y_pos = height


            plt.text(
                x_pos,
                y_pos,
                f"{height:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=6,
                color="black",
                alpha=0.0,
                clip_on=True,
            )


    for xi in np.arange(num_labels + 1) - 0.5:
        plt.axvline(xi, color="lightgray", linestyle=":", linewidth=0.6, alpha=0.7)

    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, pivot.index, rotation=45, ha="right", fontsize=8)
    plt.ylabel(f"{metric.upper()}(C3) - {metric.upper()}(C0)")
    plt.title(
        f"C3-C0 {metric.upper()} delta per label\n"
        f"(bars grouped by model; higher = more benefit from context)"
    )
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.legend(loc="upper right")
    plt.tight_layout()

    suffix = f"_top{top_k_labels}" if top_k_labels > 0 else "_all"
    png_path = os.path.join(
        out_dir, f"bar_delta_{metric}_C3_minus_C0_by_label{suffix}.png"
    )
    pdf_path = os.path.join(
        out_dir, f"bar_delta_{metric}_C3_minus_C0_by_label{suffix}.pdf"
    )
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved bar plot: {png_path}")
    print(f"Saved bar plot (PDF, selectable text): {pdf_path}")


def main() -> None:
    args = parse_args()

    df_delta = load_c0_c3_deltas(args.summary_csv, args.metric)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_heatmap(df_delta, args.metric, args.out_dir)
    plot_top_labels_bar(df_delta, args.metric, args.out_dir, args.top_k_labels)


if __name__ == "__main__":
    main()
