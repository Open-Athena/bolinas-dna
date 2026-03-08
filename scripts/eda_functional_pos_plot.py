"""EDA functional-pos: Scatter plots of log-likelihood metrics vs downstream AUPRC.

Fetches training runs from the W&B `eda-functional-pos` group, which trains gLMs of
3 sizes (6M/60M/600M) on 3 timescales (primates/mammals/vertebrates) and tracks
functional vs non-functional validation loss. Plots AUPRC against four log-likelihood
metrics: LL(all), LL(functional), LL(non-functional), and LL(functional) - LL(non-functional).

Usage:
    uv run python scripts/eda_functional_pos_plot.py
    uv run python scripts/eda_functional_pos_plot.py --refresh
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
WANDB_GROUP = "eda-functional-pos"
RUN_NAME_PREFIX = "eda-functional-pos-"

LOSS_KEY = "eval/loss"
FUNCTIONAL_LOSS_KEY = "eval/val_functional/loss"
NONFUNCTIONAL_LOSS_KEY = "eval/val_nonfunctional/loss"

TASK_PREFIX = "lm_eval/traitgym_mendelian_v2_255"
TASKS: dict[str, str] = {
    "tss_proximal": "Promoter VEP AUPRC",
    "5_prime_UTR_variant": "5' UTR VEP AUPRC",
}

OUTPUT_DIR = Path("results/eda_functional_pos")
CACHE_PATH = OUTPUT_DIR / "wandb_cache.csv"

DATASET_ORDER = ["primates", "mammals", "vertebrates"]
DATASET_MARKERS = {"primates": "o", "mammals": "^", "vertebrates": "s"}

MODEL_SIZE_ORDER = ["6M", "60M", "600M"]
MODEL_SIZE_POINTS: dict[str, int] = {"6M": 30, "60M": 80, "600M": 180}

LL_ALL = "LL(all)"
LL_FUNC = "LL(functional)"
LL_NONFUNC = "LL(non-functional)"
LL_DIFF = "LL(functional) - LL(non-functional)"
XCOLS = [LL_ALL, LL_FUNC, LL_NONFUNC, LL_DIFF]

SUBPLOT_SIZE = 4
SIGNIFICANCE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# W&B fetching
# ---------------------------------------------------------------------------


def fetch_runs() -> pd.DataFrame:
    """Fetch all checkpoints from W&B runs, returning one row per (run, step, task)."""
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, filters={"group": WANDB_GROUP})

    auprc_keys = [f"{TASK_PREFIX}/{task}/auprc" for task in TASKS]

    records: list[dict[str, str | float | int]] = []
    for run in runs:
        name: str = run.name
        if not name.startswith(RUN_NAME_PREFIX):
            continue

        suffix = name[len(RUN_NAME_PREFIX) :]
        parts = suffix.split("-")
        # Format: {timescale}-{model_size} or {timescale}-{model_size}-{hash}
        if len(parts) < 2:
            continue
        dataset = parts[0]
        model_size = parts[1].upper()

        history = run.scan_history(
            keys=[LOSS_KEY, FUNCTIONAL_LOSS_KEY, NONFUNCTIONAL_LOSS_KEY, *auprc_keys, "_step"]
        )
        seen_steps: set[int] = set()
        for row in history:
            step = row.get("_step")
            if step is None or step in seen_steps:
                continue
            seen_steps.add(step)

            loss = row.get(LOSS_KEY)
            func_loss = row.get(FUNCTIONAL_LOSS_KEY)
            nonfunc_loss = row.get(NONFUNCTIONAL_LOSS_KEY)

            for task, auprc_key in zip(TASKS, auprc_keys):
                auprc = row.get(auprc_key)
                if auprc is None:
                    continue

                record: dict[str, str | float | int] = {
                    "dataset": dataset,
                    "model size": model_size,
                    "step": int(step),
                    "task": task,
                    "auprc": float(auprc),
                }
                if loss is not None:
                    record[LL_ALL] = -float(loss)
                if func_loss is not None:
                    record[LL_FUNC] = -float(func_loss)
                if nonfunc_loss is not None:
                    record[LL_NONFUNC] = -float(nonfunc_loss)
                if func_loss is not None and nonfunc_loss is not None:
                    record[LL_DIFF] = float(nonfunc_loss) - float(func_loss)

                records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Cache
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


# ---------------------------------------------------------------------------
# Sigmoid fitting
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray, lower: float, upper: float, k: float, x0: float) -> np.ndarray:
    """Increasing sigmoid: lower at low x, upper at high x."""
    return lower + (upper - lower) / (1 + np.exp(np.clip(-k * (x - x0), -500, 500)))


def _fit_sigmoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a sigmoid to x vs y using least squares."""
    k0 = 5.0 if np.corrcoef(x, y)[0, 1] >= 0 else -5.0
    p0 = [y.min(), y.max(), k0, float(np.median(x))]
    bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])
    popt, _ = optimize.curve_fit(_sigmoid, x, y, p0=p0, bounds=bounds, maxfev=10000)
    return popt


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _correlation_subtitle(df: pd.DataFrame, xcol: str) -> str:
    """Return a compact correlation string with one-sided p-values (positive direction)."""
    if len(df) < 3:
        return ""
    pearson_r, pearson_p_two = stats.pearsonr(df[xcol], df["auprc"])
    spearman_r, spearman_p_two = stats.spearmanr(df[xcol], df["auprc"])
    pearson_p = pearson_p_two / 2 if pearson_r > 0 else 1 - pearson_p_two / 2
    spearman_p = spearman_p_two / 2 if spearman_r > 0 else 1 - spearman_p_two / 2
    r_star = " (*)" if pearson_p < SIGNIFICANCE_THRESHOLD else ""
    rho_star = " (*)" if spearman_p < SIGNIFICANCE_THRESHOLD else ""
    return "\n".join([f"$r$ = {pearson_r:.2f}{r_star}", f"$\\rho$ = {spearman_r:.2f}{rho_star}"])


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


def _annotate_correlation(ax: plt.Axes, df: pd.DataFrame, xcol: str) -> None:
    """Add correlation annotation in the emptiest corner of the axes."""
    text = _correlation_subtitle(df, xcol)
    if not text:
        return
    x, y, ha, va = _emptiest_corner(ax)
    ax.annotate(text, xy=(x, y), xycoords="axes fraction", fontsize=9, ha=ha, va=va)


# ---------------------------------------------------------------------------
# Facet helpers
# ---------------------------------------------------------------------------

XCOL_LONG = "metric"
XVAL_LONG = "value"


def _melt_ll_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide LL columns into long format for faceting."""
    return df.melt(
        id_vars=["dataset", "model size", "step", "task", "auprc"],
        value_vars=XCOLS,
        var_name=XCOL_LONG,
        value_name=XVAL_LONG,
    ).dropna(subset=[XVAL_LONG])


def _add_facet_sigmoid_fits(g: sns.FacetGrid, long_df: pd.DataFrame) -> None:
    """Overlay a sigmoid fit on each facet panel."""
    for ax, xcol in zip(g.axes.flat, XCOLS):
        subset = long_df[long_df[XCOL_LONG] == xcol]
        if len(subset) < 4:
            continue
        x = subset[XVAL_LONG].values
        y = subset["auprc"].values
        try:
            popt = _fit_sigmoid(x, y)
            x_fit = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_fit, _sigmoid(x_fit, *popt), color="C3", linewidth=1.5)
        except Exception:
            logger.warning("Sigmoid fit failed for %s", xcol)


def _add_facet_correlations(g: sns.FacetGrid, long_df: pd.DataFrame) -> None:
    """Add per-facet correlation annotations."""
    for ax, xcol in zip(g.axes.flat, XCOLS):
        subset = long_df[long_df[XCOL_LONG] == xcol]
        # Temporarily rename for the correlation helper
        panel_df = subset.rename(columns={XVAL_LONG: xcol})
        _annotate_correlation(ax, panel_df, xcol)


def _finalize_facetgrid(g: sns.FacetGrid, output_path: Path, *, ylabel: str) -> None:
    """Apply shared styling and save a FacetGrid."""
    for ax, xcol in zip(g.axes.flat, XCOLS):
        ax.set_xlabel(xcol)
        ax.set_title("")
    g.set_ylabels(ylabel)
    g.despine()
    g.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(g.figure)
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_scatter(df: pd.DataFrame, output_path: Path, *, ylabel: str) -> None:
    """1×4 scatter: one panel per LL metric."""
    long_df = _melt_ll_columns(df)

    g = sns.relplot(
        data=long_df,
        x=XVAL_LONG,
        y="auprc",
        hue="step",
        palette="viridis",
        style="dataset",
        markers=DATASET_MARKERS,
        size="model size",
        size_order=MODEL_SIZE_ORDER,
        sizes=MODEL_SIZE_POINTS,
        col=XCOL_LONG,
        col_order=XCOLS,
        kind="scatter",
        facet_kws={"sharex": False, "sharey": True},
        height=SUBPLOT_SIZE,
        aspect=1,
        edgecolor="none",
    )
    _add_facet_sigmoid_fits(g, long_df)
    _add_facet_correlations(g, long_df)
    _finalize_facetgrid(g, output_path, ylabel=ylabel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-fetch from W&B instead of cache")
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
        plot_scatter(task_df, OUTPUT_DIR / task / "scatter.svg", ylabel=ylabel)


if __name__ == "__main__":
    main()
