import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with: pip install matplotlib"
    ) from e


CONTEXTS: Tuple[str, ...] = ("C0", "C1", "C2", "C3", "C0toC3")
DEFAULT_LABEL_ORDER: Tuple[str, ...] = (
    "No emotion",
    "Sadness",
    "Loneliness",
    "Hopelessness",
    "Anger",
    "Worthlessness",
    "Suicide intent",
    "Cognitive dysfunction",
    "Emptiness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create two combined figures that summarize label-context effects "
            "across all models with a shared legend and compact model names."
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
        help="Output directory for the combined figures.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Metric to plot (default: f1).",
    )
    parser.add_argument(
        "--run_type",
        type=str,
        default="base",
        choices=["base", "ensemble", "base+ensemble"],
        help="Which run types to include (default: base).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Number of columns for the small-multiples figure (default: 2).",
    )
    return parser.parse_args()


def simplify_model_name(model_short: str) -> str:
    prefixes = (
        "FacebookAI-",
        "google-bert-",
        "microsoft-",
    )
    for prefix in prefixes:
        if model_short.startswith(prefix):
            model_short = model_short[len(prefix) :]
            break
    if model_short.startswith("mental-mental-"):
        model_short = "mental-" + model_short[len("mental-mental-") :]
    if model_short.endswith("-uncased"):
        model_short = model_short[: -len("-uncased")]
    return model_short


def build_color_map(labels: Sequence[str]) -> Dict[str, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(labels))]
    return dict(zip(labels, colors))


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 13,
        }
    )


def figure_all_models_context_curves(
    df_base: pd.DataFrame,
    df_ensemble: pd.DataFrame,
    groups: List[Tuple[str, float, float, float]],
    labels: Sequence[str],
    contexts: Sequence[str],
    metric: str,
    out_path: Path,
    cols: int,
) -> None:

    y_values = pd.concat(
        [
            df_base[df_base["context"].isin(contexts) & df_base["label"].isin(labels)],
            df_ensemble[
                df_ensemble["context"].isin(contexts) & df_ensemble["label"].isin(labels)
            ],
        ],
        ignore_index=True,
    )[metric].to_numpy()
    y_min = float(np.nanmin(y_values)) if y_values.size else 0.0
    y_max = float(np.nanmax(y_values)) if y_values.size else 1.0
    y_pad = 0.03
    y_low = max(0.0, y_min - y_pad)
    y_high = min(1.0, y_max + y_pad)

    cols = max(1, int(cols))
    rows = (len(groups) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 8.2, rows * 5.0),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    color_map = build_color_map(labels)
    x = np.arange(len(contexts))

    for ax in axes.ravel():
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_ylim(y_low, y_high)

    handles = [
        plt.Line2D([0], [0], color=color_map[label], linewidth=2.0, label=label)
        for label in labels
    ]

    for idx, (model_short, epochs, seed, freeze_layers) in enumerate(groups):
        ax = axes[idx // cols][idx % cols]
        group_df = df_base[
            (df_base["model_short"] == model_short)
            & (df_base["epochs"] == epochs)
            & (df_base["seed"] == seed)
            & (df_base["freeze_layers"] == freeze_layers)
        ]
        agg = (
            group_df.groupby(["label", "context"])[metric]
            .mean()
            .reset_index()
        )
        pivot = agg.pivot(index="context", columns="label", values=metric)
        pivot = pivot.reindex(columns=list(labels))

        ens_values = {}
        if "C0toC3" in contexts and not df_ensemble.empty:
            group_ens_df = df_ensemble[
                (df_ensemble["model_short"] == model_short)
                & (df_ensemble["epochs"] == epochs)
                & (df_ensemble["seed"] == seed)
                & (df_ensemble["freeze_layers"] == freeze_layers)
                & (df_ensemble["context"] == "C0toC3")
            ]
            if not group_ens_df.empty:
                ens_values = (
                    group_ens_df.groupby("label")[metric].mean().to_dict()
                )

        for label in labels:
            y = np.array(
                [
                    (ens_values.get(label, np.nan) if ctx == "C0toC3" else pivot.at[ctx, label])
                    if (ctx in pivot.index and label in pivot.columns) or ctx == "C0toC3"
                    else np.nan
                    for ctx in contexts
                ],
                dtype=float,
            )
            valid = np.isfinite(y)
            if valid.sum() < 2:
                continue

            x_valid = x[valid].astype(float)
            y_valid = y[valid].astype(float)


            degree = int(min(4, max(1, len(x_valid) - 1)))
            try:
                coeffs = np.polyfit(x_valid, y_valid, degree)
                x_smooth = np.linspace(float(x_valid.min()), float(x_valid.max()), 200)
                y_smooth = np.polyval(coeffs, x_smooth)
                ax.plot(
                    x_smooth,
                    y_smooth,
                    color=color_map[label],
                    linewidth=2.4,
                    alpha=0.9,
                )
                ax.scatter(
                    x_valid,
                    y_valid,
                    color=color_map[label],
                    s=18,
                    alpha=0.95,
                )
            except Exception:
                ax.plot(
                    x_valid,
                    y_valid,
                    color=color_map[label],
                    linewidth=2.2,
                    marker="o",
                    markersize=4.5,
                    alpha=0.9,
                )

        ax.set_title(simplify_model_name(model_short), fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(contexts)
        if (idx % cols) == 0:
            ax.set_ylabel(metric.upper())
        if (idx // cols) == (rows - 1):
            ax.set_xlabel("Context")


    legend_ax = None
    if len(groups) < rows * cols:
        legend_ax = axes[len(groups) // cols][len(groups) % cols]
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            title="Label",
            loc="center",
            frameon=True,
            fontsize=12,
            title_fontsize=13,
        )
        for j in range(len(groups) + 1, rows * cols):
            fig.delaxes(axes[j // cols][j % cols])
    else:
        fig.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            title="Label",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=min(5, len(labels)),
            frameon=False,
            fontsize=10,
        )

    fig.suptitle(f"Per-label {metric.upper()} vs context (all models)", y=0.98)
    fig.subplots_adjust(
        top=0.90,
        bottom=0.08,
        left=0.06,
        right=0.99,
        wspace=0.10,
        hspace=0.20,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined figure: {out_path}")


def figure_delta_heatmap(
    df: pd.DataFrame,
    groups: List[Tuple[str, float, float, float]],
    labels: Sequence[str],
    metric: str,
    out_path: Path,
) -> None:
    model_names = [simplify_model_name(m) for (m, _, _, _) in groups]
    matrix = np.zeros((len(labels), len(groups)), dtype=float)

    for j, (model_short, epochs, seed, freeze_layers) in enumerate(groups):
        group_df = df[
            (df["model_short"] == model_short)
            & (df["epochs"] == epochs)
            & (df["seed"] == seed)
            & (df["freeze_layers"] == freeze_layers)
        ]
        agg = (
            group_df.groupby(["label", "context"])[metric]
            .mean()
            .reset_index()
        )
        pivot = agg.pivot(index="label", columns="context", values=metric)
        for i, label in enumerate(labels):
            c0 = pivot.at[label, "C0"] if "C0" in pivot.columns and label in pivot.index else np.nan
            c3 = pivot.at[label, "C3"] if "C3" in pivot.columns and label in pivot.index else np.nan
            matrix[i, j] = (c3 - c0) if np.isfinite(c0) and np.isfinite(c3) else np.nan

    vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(max(7.0, len(groups) * 1.6), 5.5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"Δ {metric.upper()} (C3 − C0) by label and model")
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Δ {metric.upper()}")

    ax.set_xlabel("Model")
    ax.set_ylabel("Label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined figure: {out_path}")


def figure_delta_grouped_bars(
    df: pd.DataFrame,
    groups: List[Tuple[str, float, float, float]],
    labels: Sequence[str],
    metric: str,
    out_path: Path,
) -> None:
    model_names = [simplify_model_name(m) for (m, _, _, _) in groups]

    matrix = np.full((len(labels), len(groups)), np.nan, dtype=float)

    for j, (model_short, epochs, seed, freeze_layers) in enumerate(groups):
        group_df = df[
            (df["model_short"] == model_short)
            & (df["epochs"] == epochs)
            & (df["seed"] == seed)
            & (df["freeze_layers"] == freeze_layers)
        ]
        agg = (
            group_df.groupby(["label", "context"])[metric]
            .mean()
            .reset_index()
        )
        pivot = agg.pivot(index="label", columns="context", values=metric)
        if "C0" not in pivot.columns or "C3" not in pivot.columns:
            continue
        for i, label in enumerate(labels):
            if label not in pivot.index:
                continue
            c0 = pivot.at[label, "C0"]
            c3 = pivot.at[label, "C3"]
            if np.isfinite(c0) and np.isfinite(c3):
                matrix[i, j] = c3 - c0

    fig, ax = plt.subplots(figsize=(max(10.0, len(labels) * 1.25), 5.5))
    x = np.arange(len(labels))
    bar_width = 0.8 / max(1, len(groups))

    model_cmap = plt.get_cmap("tab10")
    for j, model_name in enumerate(model_names):
        offsets = x - 0.4 + (j + 0.5) * bar_width
        values = matrix[:, j]
        ax.bar(
            offsets,
            np.nan_to_num(values, nan=0.0),
            width=bar_width,
            label=model_name,
            color=model_cmap(j % 10),
            edgecolor="none",
        )

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(f"Δ {metric.upper()} (C3 − C0)")
    ax.set_title("C3–C0 per-label delta (grouped by model; higher = more benefit)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined figure: {out_path}")


def main() -> None:
    args = parse_args()
    apply_plot_style()
    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_file():
        raise SystemExit(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)

    df = df[df["context"].isin(CONTEXTS)].copy()
    df_base = df[df["run_type"] == "base"].copy()
    df_ensemble = df[df["run_type"] == "ensemble"].copy()

    required = {"model_short", "epochs", "seed", "freeze_layers", "label", "context", args.metric}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {', '.join(missing)}")


    group_cols = ["model_short", "epochs", "seed", "freeze_layers"]
    groups = sorted(
        df_base[group_cols].drop_duplicates().itertuples(index=False, name=None),
        key=lambda t: simplify_model_name(t[0]).lower(),
    )


    seen_labels = list(df_base["label"].dropna().unique())
    labels = [l for l in DEFAULT_LABEL_ORDER if l in seen_labels] + [
        l for l in seen_labels if l not in DEFAULT_LABEL_ORDER
    ]

    out_dir = Path(args.out_dir)
    figure_all_models_context_curves(
        df_base=df_base,
        df_ensemble=df_ensemble,
        groups=groups,
        labels=labels,
        contexts=CONTEXTS,
        metric=args.metric,
        out_path=out_dir / "all_models_context_vs_label_f1.png",
        cols=args.cols,
    )
    figure_delta_grouped_bars(
        df=df_base,
        groups=groups,
        labels=labels,
        metric=args.metric,
        out_path=out_dir / "all_models_delta_f1_C3_minus_C0_by_label.png",
    )


if __name__ == "__main__":
    main()
