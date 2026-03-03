"""EDA8: Scatter plot of validation loss vs downstream task performance (AUPRC).

Fetches training runs from W&B and plots the relationship between validation loss
and TraitGym AUPRC across checkpoints, with marker size for model size,
viridis color for training step, and markers for dataset.

Usage:
    uv run python scripts/eda8_plot.py
"""

import argparse
import logging
from pathlib import Path

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy import stats

logger = logging.getLogger(__name__)

WANDB_PROJECT = "marin"
WANDB_GROUP = "eda-ppl-vs-downstream"
LOSS_KEY = "eval/loss"
AUPRC_KEY = "lm_eval/traitgym_mendelian_v2/tss_proximal/auprc"
RUN_NAME_PREFIX = "eda-ppl-vs-downstream-"
OUTPUT_PATH = Path("results/eda8/loss_vs_auprc.svg")
CACHE_PATH = Path("results/eda8/wandb_cache.csv")

DATASET_ORDER = ["Humans", "Primates", "Mammals"]
DATASET_MARKERS = {
    "Humans": "o",
    "Primates": "^",
    "Mammals": "s",
}

MODEL_SIZE_ORDER = ["6M", "60M", "600M"]
MODEL_SIZE_POINTS: dict[str, int] = {"6M": 40, "60M": 100, "600M": 220}

SUBPLOT_SIZE = 5
XLABEL = "Validation loss"
YLABEL = "Promoter VEP AUPRC"
SIGNIFICANCE_THRESHOLD = 0.05


def fetch_runs() -> pd.DataFrame:
    """Fetch all checkpoints from W&B runs, returning one row per (run, step)."""
    api = wandb.Api()
    runs = api.runs(
        WANDB_PROJECT,
        filters={"group": WANDB_GROUP},
    )

    records: list[dict[str, str | float | int]] = []
    for run in runs:
        name: str = run.name
        if not name.startswith(RUN_NAME_PREFIX):
            continue

        # Parse: eda-ppl-vs-downstream-{dataset}-{model_size}-{hash}
        suffix = name[len(RUN_NAME_PREFIX) :]
        parts = suffix.split("-")
        if len(parts) != 3:
            continue
        dataset_raw, model_size_raw, _ = parts

        dataset = dataset_raw.capitalize()
        model_size = model_size_raw.upper()

        history = run.scan_history(keys=[LOSS_KEY, AUPRC_KEY, "_step"])
        seen_steps: dict[int, tuple[float, float]] = {}
        for row in history:
            loss = row.get(LOSS_KEY)
            auprc = row.get(AUPRC_KEY)
            step = row.get("_step")
            if loss is None or auprc is None or step is None:
                continue
            if step in seen_steps:
                prev_loss, prev_auprc = seen_steps[step]
                if (prev_loss, prev_auprc) != (float(loss), float(auprc)):
                    logger.warning(
                        "Duplicate step %d in %s with different values: "
                        "(%f, %f) vs (%f, %f)",
                        step, name, prev_loss, prev_auprc, float(loss), float(auprc),
                    )
                continue
            seen_steps[step] = (float(loss), float(auprc))

            records.append(
                {
                    "dataset": dataset,
                    "model size": model_size,
                    "step": int(step),
                    "loss": float(loss),
                    "auprc": float(auprc),
                }
            )

    return pd.DataFrame(records)


def _correlation_subtitle(df: pd.DataFrame) -> str:
    """Return a compact correlation string for use as a subtitle."""
    if len(df) < 3:
        return ""
    pearson_r, pearson_p = stats.pearsonr(df["loss"], df["auprc"])
    spearman_r, spearman_p = stats.spearmanr(df["loss"], df["auprc"])
    r_star = " (*)" if pearson_p < SIGNIFICANCE_THRESHOLD else ""
    rho_star = " (*)" if spearman_p < SIGNIFICANCE_THRESHOLD else ""
    return f"$r$ = {pearson_r:.2f}{r_star}, $\\rho$ = {spearman_r:.2f}{rho_star}"


CORNERS = [
    (0.97, 0.97, "right", "top"),
    (0.03, 0.97, "left", "top"),
    (0.97, 0.03, "right", "bottom"),
    (0.03, 0.03, "left", "bottom"),
]


def _emptiest_corner(ax: plt.Axes) -> tuple[float, float, str, str]:
    """Find the axes corner farthest from any plotted point."""
    all_offsets = []
    for child in ax.get_children():
        if isinstance(child, matplotlib.collections.PathCollection):
            offsets = child.get_offsets()
            if len(offsets) > 0:
                all_offsets.append(offsets)

    if not all_offsets:
        return CORNERS[0]

    data_points = np.concatenate(all_offsets)
    # Transform data coordinates to axes fraction
    axes_points = ax.transAxes.inverted().transform(ax.transData.transform(data_points))

    best_corner = CORNERS[0]
    best_min_dist = -1.0
    for x, y, ha, va in CORNERS:
        dists = np.sqrt((axes_points[:, 0] - x) ** 2 + (axes_points[:, 1] - y) ** 2)
        min_dist = float(dists.min())
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_corner = (x, y, ha, va)
    return best_corner


def _annotate_correlation(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Add correlation annotation in the emptiest corner of the axes."""
    text = _correlation_subtitle(df)
    if not text:
        return
    x, y, ha, va = _emptiest_corner(ax)
    ax.annotate(
        text,
        xy=(x, y),
        xycoords="axes fraction",
        fontsize=9,
        ha=ha,
        va=va,
    )


def _add_facet_correlations(
    g: sns.FacetGrid,
    df: pd.DataFrame,
    *,
    row: str | None = None,
    row_order: list[str] | None = None,
    col: str | None = None,
    col_order: list[str] | None = None,
) -> None:
    """Add per-facet correlation annotations to each axes in a FacetGrid."""
    row_vals = row_order if row is not None else [None]
    col_vals = col_order if col is not None else [None]

    for i, row_val in enumerate(row_vals):
        for j, col_val in enumerate(col_vals):
            ax = g.axes[i, j]
            if not ax.get_visible():
                continue
            mask = pd.Series(True, index=df.index)
            if row is not None:
                mask &= df[row] == row_val
            if col is not None:
                mask &= df[col] == col_val
            _annotate_correlation(ax, df[mask])


def _finalize_facetgrid(g: sns.FacetGrid, output_path: Path) -> None:
    """Apply shared styling and save a FacetGrid."""
    g.set_xlabels(XLABEL)
    g.set_ylabels(YLABEL)
    g.despine()
    g.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved plot to {output_path}")


def plot_loss_vs_auprc(df: pd.DataFrame, output_path: Path) -> None:
    """Create scatter plot of validation loss vs AUPRC (single plot, no facets)."""
    fig, ax = plt.subplots(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))

    sns.scatterplot(
        data=df,
        x="loss",
        y="auprc",
        hue="step",
        palette="viridis",
        legend="full",
        style="dataset",
        markers=DATASET_MARKERS,
        size="model size",
        size_order=MODEL_SIZE_ORDER,
        sizes=MODEL_SIZE_POINTS,
        edgecolor="none",
        ax=ax,
    )
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

    _annotate_correlation(ax, df)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    sns.despine(ax=ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_by_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """One subplot per dataset; hue=step, size=model_size."""
    g = sns.relplot(
        data=df,
        x="loss",
        y="auprc",
        hue="step",
        palette="viridis",
        legend="full",
        size="model size",
        size_order=MODEL_SIZE_ORDER,
        sizes=MODEL_SIZE_POINTS,
        col="dataset",
        col_order=DATASET_ORDER,
        kind="scatter",
        facet_kws={"sharex": False, "sharey": False},
        height=SUBPLOT_SIZE,
        aspect=1,
        edgecolor="none",
    )
    _add_facet_correlations(g, df, col="dataset", col_order=DATASET_ORDER)
    _finalize_facetgrid(g, output_path)


def plot_by_model_size(df: pd.DataFrame, output_path: Path) -> None:
    """One subplot per model size; hue=step, style=dataset."""
    g = sns.relplot(
        data=df,
        x="loss",
        y="auprc",
        hue="step",
        palette="viridis",
        legend="full",
        style="dataset",
        markers=DATASET_MARKERS,
        col="model size",
        col_order=MODEL_SIZE_ORDER,
        kind="scatter",
        facet_kws={"sharex": False, "sharey": False},
        height=SUBPLOT_SIZE,
        aspect=1,
        edgecolor="none",
    )
    _add_facet_correlations(g, df, col="model size", col_order=MODEL_SIZE_ORDER)
    _finalize_facetgrid(g, output_path)


def plot_by_run(df: pd.DataFrame, output_path: Path) -> None:
    """One subplot per run (dataset x model_size); hue=step, with margin titles."""
    g = sns.relplot(
        data=df,
        x="loss",
        y="auprc",
        hue="step",
        palette="viridis",
        legend="full",
        col="model size",
        col_order=MODEL_SIZE_ORDER,
        row="dataset",
        row_order=DATASET_ORDER,
        kind="scatter",
        facet_kws={"sharex": False, "sharey": False, "margin_titles": True},
        height=SUBPLOT_SIZE,
        aspect=1,
        edgecolor="none",
    )
    _add_facet_correlations(
        g, df,
        row="dataset", row_order=DATASET_ORDER,
        col="model size", col_order=MODEL_SIZE_ORDER,
    )
    _finalize_facetgrid(g, output_path)


def load_data(*, refresh: bool = False) -> pd.DataFrame:
    """Load data from cache if available, otherwise fetch from W&B."""
    if not refresh and CACHE_PATH.exists():
        print(f"Loading cached data from {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    df = fetch_runs()
    print(f"Fetched {len(df)} checkpoints from W&B")
    if not df.empty:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        print(f"Cached to {CACHE_PATH}")
    return df


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true", help="Re-fetch from W&B instead of cache"
    )
    args = parser.parse_args()

    df = load_data(refresh=args.refresh)
    if df.empty:
        print("No runs found — check W&B project/group settings")
        return
    print(df.to_string(index=False))
    output_dir = OUTPUT_PATH.parent
    plot_loss_vs_auprc(df, OUTPUT_PATH)
    plot_by_dataset(df, output_dir / "loss_vs_auprc_by_dataset.svg")
    plot_by_model_size(df, output_dir / "loss_vs_auprc_by_model_size.svg")
    plot_by_run(df, output_dir / "loss_vs_auprc_by_run.svg")


if __name__ == "__main__":
    main()
