"""EDA8: Scatter plot of validation log-likelihood vs downstream task performance (AUPRC).

Fetches training runs from W&B and plots the relationship between validation log-likelihood
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
RUN_NAME_PREFIX = "eda-ppl-vs-downstream-"
OUTPUT_DIR = Path("results/eda8")
CACHE_PATH = Path("results/eda8/wandb_cache.csv")

TASK_PREFIX = "lm_eval/traitgym_mendelian_v2_255"
TASKS: dict[str, str] = {
    "tss_proximal": "Promoter VEP AUPRC",
    "5_prime_UTR_variant": "5' UTR VEP AUPRC",
}

LEAKAGE_FILTER_MAP = {
    "id1-cov1": "no filter",
    "id0.3-cov0.3": "leakage filter",
}
LEAKAGE_FILTER_ORDER = ["no filter", "leakage filter"]

DATASET_ORDER = ["primates", "mammals", "vertebrates"]
DATASET_MARKERS = {"primates": "o", "mammals": "^", "vertebrates": "s"}

MODEL_SIZE_ORDER = ["6M", "60M", "600M"]
MODEL_SIZE_POINTS: dict[str, int] = {"6M": 30, "60M": 80, "600M": 180}

SUBPLOT_SIZE = 4
XCOL = "log-likelihood"
XLABEL = "Validation log-likelihood"
SIGNIFICANCE_THRESHOLD = 0.05


def fetch_runs() -> pd.DataFrame:
    """Fetch all checkpoints from W&B runs, returning one row per (run, step, task)."""
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

        # Only process runs launched after March 4th 2026
        if run.created_at < "2026-03-04":
            continue

        # Format: {dataset}-{id_thresh}-{cov_thresh}-{model_size}-{hash}
        suffix = name[len(RUN_NAME_PREFIX) :]
        parts = suffix.split("-")
        if len(parts) != 5:
            continue

        dataset = parts[0]
        version_key = f"{parts[1]}-{parts[2]}"
        model_size = parts[3].upper()
        leakage_filter = LEAKAGE_FILTER_MAP.get(version_key)
        if leakage_filter is None:
            logger.warning("Unknown version key %s in run %s", version_key, name)
            continue

        auprc_keys = [f"{TASK_PREFIX}/{task}/auprc" for task in TASKS]
        history = run.scan_history(keys=[LOSS_KEY, *auprc_keys, "_step"])
        seen_steps: set[int] = set()
        for row in history:
            loss = row.get(LOSS_KEY)
            step = row.get("_step")
            if loss is None or step is None:
                continue
            if step in seen_steps:
                continue
            seen_steps.add(step)

            for task, auprc_key in zip(TASKS, auprc_keys):
                auprc = row.get(auprc_key)
                if auprc is None:
                    continue
                records.append(
                    {
                        "dataset": dataset,
                        "leakage filter": leakage_filter,
                        "model size": model_size,
                        "step": int(step),
                        XCOL: -float(loss),
                        "auprc": float(auprc),
                        "task": task,
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _correlation_subtitle(df: pd.DataFrame, *, r2: float | None = None) -> str:
    """Return a compact correlation string with one-sided p-values (positive direction)."""
    if len(df) < 3:
        return ""
    pearson_r, pearson_p_two = stats.pearsonr(df[XCOL], df["auprc"])
    spearman_r, spearman_p_two = stats.spearmanr(df[XCOL], df["auprc"])
    # One-sided test for positive correlation: halve p-value if r > 0, else 1 - p/2
    pearson_p = pearson_p_two / 2 if pearson_r > 0 else 1 - pearson_p_two / 2
    spearman_p = spearman_p_two / 2 if spearman_r > 0 else 1 - spearman_p_two / 2
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


# ---------------------------------------------------------------------------
# Facet helpers
# ---------------------------------------------------------------------------

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
            x = subset[XCOL].values
            auprc = subset["auprc"].values
            try:
                popt = _fit_sigmoid(x, auprc)
                r2_map[(i, j)] = _r_squared(auprc, _sigmoid(x, *popt))
                x_fit = np.linspace(x.min(), x.max(), 200)
                ax.plot(x_fit, _sigmoid(x_fit, *popt), color="C3", linewidth=1.5)
            except Exception:
                logger.warning("Sigmoid fit failed for facet row=%s col=%s", row_val, col_val)

    return r2_map


def _finalize_facetgrid(g: sns.FacetGrid, output_path: Path, *, ylabel: str) -> None:
    """Apply shared styling and save a FacetGrid."""
    g.set_xlabels(XLABEL)
    g.set_ylabels(ylabel)
    g.despine()
    g.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_by_leakage_filter(df: pd.DataFrame, output_path: Path, *, ylabel: str) -> None:
    """Two panels: left = no filter, right = leakage filter."""
    g = sns.relplot(
        data=df,
        x=XCOL,
        y="auprc",
        hue="step",
        palette="viridis",
        style="dataset",
        markers=DATASET_MARKERS,
        size="model size",
        size_order=MODEL_SIZE_ORDER,
        sizes=MODEL_SIZE_POINTS,
        col="leakage filter",
        col_order=LEAKAGE_FILTER_ORDER,
        kind="scatter",
        facet_kws={"sharex": False, "sharey": False},
        height=SUBPLOT_SIZE,
        aspect=1,
        edgecolor="none",
    )
    r2_map = _add_facet_sigmoid_fits(
        g, df, col="leakage filter", col_order=LEAKAGE_FILTER_ORDER,
    )
    _add_facet_correlations(
        g, df, col="leakage filter", col_order=LEAKAGE_FILTER_ORDER, r2_map=r2_map,
    )
    _finalize_facetgrid(g, output_path, ylabel=ylabel)


def plot_by_run(df: pd.DataFrame, output_path: Path, *, ylabel: str) -> None:
    """Row=dataset, col=model size. One subplot per individual run."""
    g = sns.relplot(
        data=df,
        x=XCOL,
        y="auprc",
        hue="step",
        palette="viridis",
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
    _finalize_facetgrid(g, output_path, ylabel=ylabel)


def plot_by_dataset_and_leakage_filter(
    df: pd.DataFrame, output_path: Path, *, ylabel: str,
) -> None:
    """Row=dataset, col=leakage filter. Marker size for model size, viridis for step."""
    g = sns.relplot(
        data=df,
        x=XCOL,
        y="auprc",
        hue="step",
        palette="viridis",
        size="model size",
        size_order=MODEL_SIZE_ORDER,
        sizes=MODEL_SIZE_POINTS,
        col="leakage filter",
        col_order=LEAKAGE_FILTER_ORDER,
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
        col="leakage filter", col_order=LEAKAGE_FILTER_ORDER,
    )
    _add_facet_correlations(
        g, df,
        row="dataset", row_order=DATASET_ORDER,
        col="leakage filter", col_order=LEAKAGE_FILTER_ORDER,
        r2_map=r2_map,
    )
    _finalize_facetgrid(g, output_path, ylabel=ylabel)


# ---------------------------------------------------------------------------
# Sigmoid fitting
# ---------------------------------------------------------------------------

def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute coefficient of determination (R²)."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    return 1 - ss_res / ss_tot


def _sigmoid(x: np.ndarray, lower: float, upper: float, k: float, x0: float) -> np.ndarray:
    """Increasing sigmoid: lower at low x, upper at high x."""
    return lower + (upper - lower) / (1 + np.exp(np.clip(-k * (x - x0), -500, 500)))


def _fit_sigmoid(x: np.ndarray, auprc: np.ndarray) -> np.ndarray:
    """Fit a sigmoid to x vs AUPRC using robust regression."""
    # Choose initial k sign based on data trend
    k0 = 5.0 if np.corrcoef(x, auprc)[0, 1] >= 0 else -5.0
    p0 = np.array([auprc.min(), auprc.max(), k0, float(np.median(x))])
    bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])

    def residuals(params: np.ndarray) -> np.ndarray:
        return _sigmoid(x, *params) - auprc

    result = optimize.least_squares(
        residuals, p0, bounds=bounds, loss="soft_l1", max_nfev=10000,
    )
    return result.x


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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

    for task, ylabel in TASKS.items():
        task_df = df[df["task"] == task]
        if task_df.empty:
            print(f"No data for task {task}")
            continue
        task_dir = OUTPUT_DIR / task
        plot_by_leakage_filter(task_df, task_dir / "by_leakage_filter.svg", ylabel=ylabel)
        plot_by_dataset_and_leakage_filter(
            task_df, task_dir / "by_dataset_and_leakage_filter.svg", ylabel=ylabel,
        )
        for lf in LEAKAGE_FILTER_ORDER:
            lf_df = task_df[task_df["leakage filter"] == lf]
            if lf_df.empty:
                continue
            slug = lf.replace(" ", "_")
            plot_by_run(lf_df, task_dir / f"by_run_{slug}.svg", ylabel=ylabel)


if __name__ == "__main__":
    main()
