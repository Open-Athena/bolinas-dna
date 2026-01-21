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


def _draw_baseline_lines(
    ax: plt.Axes,
    baselines: dict[str, float],
) -> None:
    """Draw horizontal dashed lines with text labels for baseline model performance.

    Args:
        ax: Matplotlib axes to draw on.
        baselines: Dictionary mapping baseline model names to their metric values.
    """
    if not baselines:
        return

    sorted_baselines = sorted(baselines.items(), key=lambda x: x[1])

    x_max = ax.get_xlim()[1]
    x_pos = x_max * 0.98

    min_separation = 0.01
    prev_y = None
    offset_count = 0

    for name, value in sorted_baselines:
        ax.axhline(y=value, linestyle="--", color="gray", alpha=0.7, linewidth=1)

        if prev_y is not None and abs(value - prev_y) < min_separation:
            offset_count += 1
        else:
            offset_count = 0
        prev_y = value

        y_offset = offset_count * min_separation * 0.5
        ax.text(
            x_pos,
            value + y_offset,
            name,
            fontsize=7,
            color="gray",
            va="bottom",
            ha="right",
        )


def plot_models_comparison(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    score_type: str | None = None,
    dataset_subset_score_map: dict[tuple[str, str], str] | None = None,
    figsize: tuple[int, int] = (15, 10),
    models_filter: list[str] | None = None,
    dataset_subsets_filter: list[tuple[str, str]] | None = None,
    baseline_data: dict[tuple[str, str], dict[str, float]] | None = None,
) -> None:
    """Plot metric values vs training step comparing models for a specific score type.

    Creates a grid of subplots where each panel shows a (dataset, subset) combination.
    Each panel has separate lines for each model.

    Args:
        metrics_df: DataFrame with columns [step, dataset, metric, score_type, subset, value, model].
        output_path: Path where the plot will be saved (SVG).
        score_type: Score type to plot for all subplots. Mutually exclusive with dataset_subset_score_map.
        dataset_subset_score_map: Mapping of (dataset, subset) -> score_type for per-subplot score types.
            Mutually exclusive with score_type.
        figsize: Figure size in inches (width, height).
        models_filter: If provided, only include these models. None means include all.
        dataset_subsets_filter: If provided, only include these (dataset, subset) pairs.
            None means include all. Format: [(dataset1, subset1), (dataset2, subset2), ...]
        baseline_data: Optional mapping of (dataset, subset) -> {baseline_name: metric_value}
            for drawing horizontal reference lines.
    """
    output_path = Path(output_path)

    # Validate that exactly one of score_type or dataset_subset_score_map is provided
    if score_type is None and dataset_subset_score_map is None:
        msg = "Must provide either score_type or dataset_subset_score_map"
        raise ValueError(msg)
    if score_type is not None and dataset_subset_score_map is not None:
        msg = "Cannot provide both score_type and dataset_subset_score_map"
        raise ValueError(msg)

    # Apply global filters first
    filtered_df = metrics_df.copy()

    if models_filter is not None:
        filtered_df = filtered_df[filtered_df["model"].isin(models_filter)]

    # Handle score_type filtering
    if score_type is not None:
        # Single score type for all subplots
        filtered_df = filtered_df[filtered_df["score_type"] == score_type]
        if filtered_df.empty:
            msg = f"No data found for score_type: {score_type}"
            raise ValueError(msg)

        # Apply dataset_subsets_filter if provided
        if dataset_subsets_filter is not None:
            mask = pd.Series(False, index=filtered_df.index)
            for dataset, subset in dataset_subsets_filter:
                mask |= (filtered_df["dataset"] == dataset) & (
                    filtered_df["subset"] == subset
                )
            filtered_df = filtered_df[mask]

        if filtered_df.empty:
            msg = "No data found after applying filters"
            raise ValueError(msg)

        # Get unique (dataset, subset) combinations
        combinations = filtered_df[["dataset", "subset"]].drop_duplicates()
    else:
        # Different score type per subplot
        # Get combinations from the map
        combinations_list = list(dataset_subset_score_map.keys())
        combinations = pd.DataFrame(combinations_list, columns=["dataset", "subset"])

        # Filter to only include the (dataset, subset) pairs in the map
        mask = pd.Series(False, index=filtered_df.index)
        for dataset, subset in combinations_list:
            mask |= (filtered_df["dataset"] == dataset) & (
                filtered_df["subset"] == subset
            )
        filtered_df = filtered_df[mask]

        if filtered_df.empty:
            msg = "No data found after applying filters"
            raise ValueError(msg)

    # Sort by dataset, then put "global" first within each dataset
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

        # Determine which score type to use for this subplot
        if dataset_subset_score_map is not None:
            subplot_score_type = dataset_subset_score_map[(dataset, subset)]
        else:
            subplot_score_type = score_type

        # Filter data for this combination
        subplot_data = filtered_df[
            (filtered_df["dataset"] == dataset) & (filtered_df["subset"] == subset)
        ]

        # If using per-subplot score types, filter by the appropriate score type
        if dataset_subset_score_map is not None:
            subplot_data = subplot_data[
                subplot_data["score_type"] == subplot_score_type
            ]

        # Use first metric for this dataset
        if not subplot_data.empty:
            primary_metric = subplot_data["metric"].iloc[0]
            plot_data = subplot_data[subplot_data["metric"] == primary_metric]
        else:
            continue

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

        if baseline_data is not None:
            key = (dataset, subset)
            if key in baseline_data:
                _draw_baseline_lines(ax, baseline_data[key])

        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"{primary_metric}")

        # Include score type in title if using per-subplot score types
        if dataset_subset_score_map is not None:
            ax.set_title(f"{dataset}\n{subset}\n{subplot_score_type}", fontsize=10)
        else:
            ax.set_title(f"{dataset}\n{subset}", fontsize=10)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    # Set main title based on mode
    if score_type is not None:
        plt.suptitle(f"Score Type: {score_type}", fontsize=14, y=0.995)
    else:
        plt.suptitle("Model Comparison (Multiple Score Types)", fontsize=14, y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
