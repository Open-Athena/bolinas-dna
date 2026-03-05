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
from scipy import optimize, stats

logger = logging.getLogger(__name__)

WANDB_PROJECT = "marin"
WANDB_GROUP = "eda-ppl-vs-downstream"
LOSS_KEY = "eval/loss"
AUPRC_KEY = "lm_eval/traitgym_mendelian_v2_255/tss_proximal/auprc"
RUN_NAME_PREFIX = "eda-ppl-vs-downstream-"
OUTPUT_PATH = Path("results/eda8/loss_vs_auprc.svg")
CACHE_PATH = Path("results/eda8/wandb_cache.csv")

DATASET_ORDER = [
    "no-leakage-filter",
    "leakage-filter",
]
DATASET_MARKERS = {
    "no-leakage-filter": "o",
    "leakage-filter": "s",
}

MODEL_SIZE_ORDER = ["60M", "600M"]
MODEL_SIZE_POINTS: dict[str, int] = {"60M": 60, "600M": 160}

SUBPLOT_SIZE = 4
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

        # Parse run name suffix after prefix
        # Old format: {dataset}-{model_size}-{hash}
        # New format: {dataset}-{version}-{model_size}-{hash}
        suffix = name[len(RUN_NAME_PREFIX) :]
        parts = suffix.split("-")

        # Only process runs launched after March 4th 2025
        if run.created_at < "2026-03-04":
            continue

        dataset_version_map = {
            "id1-cov1": "no-leakage-filter",
            "id0.3-cov0.3": "leakage-filter",
        }

        if len(parts) == 5:
            dataset_raw = parts[0]
            version_key = f"{parts[1]}-{parts[2]}"
            model_size_raw = parts[3]
            dataset_version = dataset_version_map.get(version_key, version_key)
        elif len(parts) == 3:
            dataset_raw = parts[0]
            model_size_raw = parts[1]
            dataset_version = None
        else:
            continue

        if dataset_version:
            dataset = dataset_version
        else:
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


def _correlation_subtitle(df: pd.DataFrame, *, r2: float | None = None) -> str:
    """Return a compact correlation string for use as a subtitle."""
    if len(df) < 3:
        return ""
    pearson_r, pearson_p = stats.pearsonr(df["loss"], df["auprc"])
    spearman_r, spearman_p = stats.spearmanr(df["loss"], df["auprc"])
    r_star = " (*)" if pearson_p < SIGNIFICANCE_THRESHOLD else ""
    rho_star = " (*)" if spearman_p < SIGNIFICANCE_THRESHOLD else ""
    lines = [f"$r$ = {pearson_r:.2f}{r_star}", f"$\\rho$ = {spearman_r:.2f}{rho_star}"]
    if r2 is not None:
        lines.append(f"sigmoid $R^2$ = {r2:.2f}")
    return "\n".join(lines)


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


def _annotate_correlation(ax: plt.Axes, df: pd.DataFrame, *, r2: float | None = None) -> None:
    """Add correlation annotation in the emptiest corner of the axes."""
    text = _correlation_subtitle(df, r2=r2)
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


def _facet_mask(
    df: pd.DataFrame,
    row: str | None,
    row_val: str | None,
    col: str | None,
    col_val: str | None,
) -> pd.Series:
    """Build a boolean mask for a single facet cell."""
    mask = pd.Series(True, index=df.index)
    if row is not None:
        mask &= df[row] == row_val
    if col is not None:
        mask &= df[col] == col_val
    return mask


def _add_facet_correlations(
    g: sns.FacetGrid,
    df: pd.DataFrame,
    *,
    row: str | None = None,
    row_order: list[str] | None = None,
    col: str | None = None,
    col_order: list[str] | None = None,
    r2_map: dict[tuple[int, int], float] | None = None,
) -> None:
    """Add per-facet correlation annotations to each axes in a FacetGrid."""
    row_vals = row_order if row is not None else [None]
    col_vals = col_order if col is not None else [None]

    for i, row_val in enumerate(row_vals):
        for j, col_val in enumerate(col_vals):
            ax = g.axes[i, j]
            if not ax.get_visible():
                continue
            mask = _facet_mask(df, row, row_val, col, col_val)
            r2 = r2_map.get((i, j)) if r2_map is not None else None
            _annotate_correlation(ax, df[mask], r2=r2)


def _add_facet_sigmoid_fits(
    g: sns.FacetGrid,
    df: pd.DataFrame,
    *,
    row: str | None = None,
    row_order: list[str] | None = None,
    col: str | None = None,
    col_order: list[str] | None = None,
) -> dict[tuple[int, int], float]:
    """Overlay a sigmoid fit on each facet. Returns R² per (i, j) facet index."""
    row_vals = row_order if row is not None else [None]
    col_vals = col_order if col is not None else [None]
    r2_map: dict[tuple[int, int], float] = {}

    for i, row_val in enumerate(row_vals):
        for j, col_val in enumerate(col_vals):
            ax = g.axes[i, j]
            if not ax.get_visible():
                continue
            mask = _facet_mask(df, row, row_val, col, col_val)
            subset = df[mask]
            if len(subset) < 4:
                continue
            loss = subset["loss"].values
            auprc = subset["auprc"].values
            try:
                popt = _fit_sigmoid(loss, auprc)
                r2_map[(i, j)] = _r_squared(auprc, _sigmoid(loss, *popt))
                x_fit = np.linspace(loss.min(), loss.max(), 200)
                ax.plot(x_fit, _sigmoid(x_fit, *popt), color="C3", linewidth=1.5)
            except Exception:
                logger.warning("Sigmoid fit failed for facet row=%s col=%s", row_val, col_val)

    return r2_map


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
    g = sns.relplot(
        data=df,
        x="loss",
        y="auprc",
        hue="step",
        palette="viridis",
        style="dataset",
        markers=DATASET_MARKERS,
        size="model size",
        size_order=MODEL_SIZE_ORDER,
        sizes=MODEL_SIZE_POINTS,
        kind="scatter",
        height=SUBPLOT_SIZE,
        aspect=1,
        edgecolor="none",
    )
    _annotate_correlation(g.ax, df)
    _finalize_facetgrid(g, output_path)


def plot_by_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """One subplot per dataset; hue=step, style=dataset, size=model_size."""
    g = sns.relplot(
        data=df,
        x="loss",
        y="auprc",
        hue="step",
        palette="viridis",
        style="dataset",
        markers=DATASET_MARKERS,
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
    r2_map = _add_facet_sigmoid_fits(g, df, col="dataset", col_order=DATASET_ORDER)
    _add_facet_correlations(g, df, col="dataset", col_order=DATASET_ORDER, r2_map=r2_map)
    _finalize_facetgrid(g, output_path)



def plot_by_run(df: pd.DataFrame, output_path: Path) -> None:
    """One subplot per run (dataset x model_size); hue=step, style=dataset, size=model_size."""
    g = sns.relplot(
        data=df,
        x="loss",
        y="auprc",
        hue="step",
        palette="viridis",
        style="dataset",
        markers=DATASET_MARKERS,
        size="model size",
        size_order=MODEL_SIZE_ORDER,
        sizes=MODEL_SIZE_POINTS,
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
    r2_map = _add_facet_sigmoid_fits(
        g, df,
        row="dataset", row_order=DATASET_ORDER,
        col="model size", col_order=MODEL_SIZE_ORDER,
    )
    _add_facet_correlations(
        g, df,
        row="dataset", row_order=DATASET_ORDER,
        col="model size", col_order=MODEL_SIZE_ORDER,
        r2_map=r2_map,
    )
    _finalize_facetgrid(g, output_path)


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute coefficient of determination (R²)."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    return 1 - ss_res / ss_tot


def _sigmoid(loss: np.ndarray, lower: float, upper: float, k: float, x0: float) -> np.ndarray:
    """Decreasing sigmoid: upper at low loss, lower at high loss."""
    return lower + (upper - lower) / (1 + np.exp(k * (loss - x0)))


def _fit_sigmoid(loss: np.ndarray, auprc: np.ndarray) -> np.ndarray:
    """Fit a sigmoid to loss vs AUPRC using robust regression, returning parameters."""
    p0 = np.array([auprc.min(), auprc.max(), 5.0, float(np.median(loss))])
    bounds = ([0, 0, 0, -np.inf], [1, 1, np.inf, np.inf])

    def residuals(params: np.ndarray) -> np.ndarray:
        return _sigmoid(loss, *params) - auprc

    result = optimize.least_squares(
        residuals, p0, bounds=bounds, loss="soft_l1", max_nfev=10000,
    )
    return result.x



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

    plot_by_dataset(df, output_dir / "loss_vs_auprc_by_dataset.svg")

    plot_by_run(df, output_dir / "loss_vs_auprc_by_run.svg")


if __name__ == "__main__":
    main()
