"""Common imports and helper functions for sequence similarity analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from datasets import load_dataset


def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a specific dataset."""
    for dataset in config["datasets"]:
        if dataset["name"] == dataset_name:
            return dataset
    raise ValueError(f"Dataset {dataset_name} not found in config")


def get_hf_path(dataset_name: str) -> str:
    """Get HuggingFace path for a dataset."""
    return get_dataset_config(dataset_name)["hf_path"]


def load_sequences_from_hf(
    hf_path: str,
    split: str,
    seq_column: str = "seq",
) -> pl.DataFrame:
    """Load sequences from a HuggingFace dataset.

    Reverse complement rows (id ending with ``_-``) are filtered out because
    mmseqs2 ``--strand 2`` searches both strands automatically.

    Args:
        hf_path: HuggingFace dataset path
        split: Dataset split (train or validation)
        seq_column: Name of the sequence column

    Returns:
        Polars DataFrame with columns [id, seq, split]
    """
    ds = load_dataset(hf_path, split=split)

    ids = ds["id"]
    sequences = ds[seq_column]

    df = pl.DataFrame({
        "id": ids,
        "seq": sequences,
    })

    # Filter out reverse complement rows (added by add_rc() during dataset creation)
    total_before = df.height
    df = df.filter(~pl.col("id").str.ends_with("_-"))
    n_filtered = total_before - df.height
    if n_filtered > 0:
        print(f"  Filtered {n_filtered:,} reverse complement rows ({n_filtered / total_before * 100:.1f}%) from {split}")

    df = df.with_columns(pl.lit(split).alias("split"))
    return df.select(["id", "seq", "split"])


GENOME_SET_SPECIES = {
    "humans": 1,
    "primates": 11,
    "mammals": 81,
}


def _plot_train_matches(summary_path, output_path, col_name, label, fmt):
    """Plot a single heatmap: rows = coverage × region, cols = genome sets."""
    import numpy as np

    df = pl.read_parquet(summary_path).to_pandas()
    dataset_order = [d["name"] for d in config["datasets"]]
    datasets = [d for d in dataset_order if d in df["dataset"].values]

    def parse_dataset(name):
        return name.rsplit("_", 1)  # (genome_set, interval_type)

    genome_sets = list(dict.fromkeys(parse_dataset(d)[0] for d in datasets))
    interval_types = list(dict.fromkeys(parse_dataset(d)[1] for d in datasets))
    identity_thresholds = sorted(df["identity_threshold"].unique())
    coverage_thresholds = sorted(df["coverage_threshold"].unique())

    # Build matrix: rows = (region, coverage), cols = (genome_set, identity)
    n_cov = len(coverage_thresholds)
    n_id = len(identity_thresholds)
    n_rows = len(interval_types) * n_cov
    n_cols = len(genome_sets) * n_id
    matrix = np.full((n_rows, n_cols), np.nan)

    row_labels = []
    for interval_type in interval_types:
        for cov in coverage_thresholds:
            row_labels.append(str(cov))

    for i, interval_type in enumerate(interval_types):
        for j, cov in enumerate(coverage_thresholds):
            row = i * n_cov + j
            for k, genome_set in enumerate(genome_sets):
                for m, ident in enumerate(identity_thresholds):
                    col = k * n_id + m
                    dataset_name = f"{genome_set}_{interval_type}"
                    mask = (
                        (df["dataset"] == dataset_name)
                        & (df["identity_threshold"] == ident)
                        & (df["coverage_threshold"] == cov)
                    )
                    vals = df.loc[mask, col_name]
                    if len(vals) > 0:
                        matrix[row, col] = vals.iloc[0]

    # Sub-column labels: identity thresholds repeated per genome set
    col_labels = [str(ident) for _ in genome_sets for ident in identity_thresholds]

    fig, ax = plt.subplots(figsize=(1.4 * n_cols + 1.5, 0.6 * n_rows + 2))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="YlOrRd",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={
            "label": f"{label} train matches",
            "orientation": "horizontal",
            "shrink": 0.5,
            "pad": 0.15,
        },
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )

    # Identity ticks at the bottom, genome set headers at the top
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position("bottom")
    ax.set_xlabel("Identity threshold")
    ax.set_ylabel("Coverage threshold")

    # Add genome set group labels at the top
    for k, gs in enumerate(genome_sets):
        n_sp = GENOME_SET_SPECIES.get(gs, "?")
        sp_label = "1 species" if n_sp == 1 else f"{n_sp} species"
        center_x = k * n_id + n_id / 2
        ax.text(
            center_x, -0.4, f"{gs}\n({sp_label})",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            transform=ax.transData,
        )

    # Draw vertical separators between genome set groups
    for k in range(1, len(genome_sets)):
        x = k * n_id
        ax.axvline(x, color="black", linewidth=2)

    # Add region labels on the right via a secondary y-axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([
        i * n_cov + n_cov / 2
        for i in range(len(interval_types))
    ])
    ax2.set_yticklabels(interval_types, fontsize=12, fontweight="bold")
    ax2.tick_params(right=False, pad=15)

    # Draw horizontal separator between region groups
    for i in range(1, len(interval_types)):
        y = i * n_cov
        ax.axhline(y, color="black", linewidth=2)

    fig.suptitle(f"{label} Train Matches per Validation Sequence", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")
