"""Plotting utilities for visualizing evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics_vs_step(
    metrics_df: pd.DataFrame,
    model_name: str,
    output_path: str | Path,
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """Plot metric values vs training step for a specific model.

    Creates a grid of subplots where each panel shows a (dataset, subset) combination.
    Each panel has separate lines for each scoring method (minus_llr, abs_llr).

    Args:
        metrics_df: DataFrame with columns [step, dataset, metric, score_type, subset, value].
        model_name: Name of the model to plot (used for filtering and title).
        output_path: Path where the plot will be saved (SVG).
        figsize: Figure size in inches (width, height).
    """
    output_path = Path(output_path)

    # Get unique (dataset, subset) combinations
    # Sort by dataset, then put "global" first within each dataset
    combinations = metrics_df[["dataset", "subset"]].drop_duplicates()
    combinations["sort_key"] = combinations["subset"].apply(
        lambda x: (0, x) if x == "global" else (1, x)
    )
    combinations = combinations.sort_values(["dataset", "sort_key"]).drop(
        columns=["sort_key"]
    )

    # Calculate grid dimensions
    n_plots = len(combinations)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (_, row) in enumerate(combinations.iterrows()):
        dataset = row["dataset"]
        subset = row["subset"]

        # Filter data for this combination (use first metric for this dataset)
        dataset_data = metrics_df[metrics_df["dataset"] == dataset]
        primary_metric = dataset_data["metric"].iloc[0]

        plot_data = metrics_df[
            (metrics_df["dataset"] == dataset)
            & (metrics_df["subset"] == subset)
            & (metrics_df["metric"] == primary_metric)
        ]

        # Plot each score type
        ax = axes[idx]
        for score_type in sorted(plot_data["score_type"].unique()):
            score_data = plot_data[plot_data["score_type"] == score_type]
            ax.plot(
                score_data["step"],
                score_data["value"],
                marker="o",
                label=score_type,
                linewidth=2,
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"{primary_metric}")
        ax.set_title(f"{dataset}\n{subset}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Model: {model_name}", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_models_comparison(
    metrics_df: pd.DataFrame,
    score_type: str,
    output_path: str | Path,
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """Plot metric values vs training step comparing all models for a specific score type.

    Creates a grid of subplots where each panel shows a (dataset, subset) combination.
    Each panel has separate lines for each model.

    Args:
        metrics_df: DataFrame with columns [step, dataset, metric, score_type, subset, value, model].
        score_type: Score type to plot (e.g., minus_llr, abs_llr, embed_last_l2).
        output_path: Path where the plot will be saved (SVG).
        figsize: Figure size in inches (width, height).
    """
    output_path = Path(output_path)

    # Filter by score type
    filtered_df = metrics_df[metrics_df["score_type"] == score_type]

    if filtered_df.empty:
        msg = f"No data found for score_type: {score_type}"
        raise ValueError(msg)

    # Get unique (dataset, subset) combinations
    # Sort by dataset, then put "global" first within each dataset
    combinations = filtered_df[["dataset", "subset"]].drop_duplicates()
    combinations["sort_key"] = combinations["subset"].apply(
        lambda x: (0, x) if x == "global" else (1, x)
    )
    combinations = combinations.sort_values(["dataset", "sort_key"]).drop(
        columns=["sort_key"]
    )

    # Calculate grid dimensions
    n_plots = len(combinations)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (_, row) in enumerate(combinations.iterrows()):
        dataset = row["dataset"]
        subset = row["subset"]

        # Filter data for this combination (use first metric for this dataset)
        dataset_data = filtered_df[filtered_df["dataset"] == dataset]
        primary_metric = dataset_data["metric"].iloc[0]

        plot_data = filtered_df[
            (filtered_df["dataset"] == dataset)
            & (filtered_df["subset"] == subset)
            & (filtered_df["metric"] == primary_metric)
        ]

        # Plot each model
        ax = axes[idx]
        for model in sorted(plot_data["model"].unique()):
            model_data = plot_data[plot_data["model"] == model]
            ax.plot(
                model_data["step"],
                model_data["value"],
                marker="o",
                label=model,
                linewidth=2,
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"{primary_metric}")
        ax.set_title(f"{dataset}\n{subset}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Score Type: {score_type}", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
