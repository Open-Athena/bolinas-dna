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
AUPRC_KEY = "lm_eval/traitgym_mendelian_v2_255/tss_proximal/auprc"

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
    """Fetch all checkpoints from W&B runs, returning one row per (run, step)."""
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, filters={"group": WANDB_GROUP})

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
            keys=[LOSS_KEY, FUNCTIONAL_LOSS_KEY, NONFUNCTIONAL_LOSS_KEY, AUPRC_KEY, "_step"]
        )
        seen_steps: set[int] = set()
        for row in history:
            step = row.get("_step")
            auprc = row.get(AUPRC_KEY)
            if step is None or auprc is None:
                continue
            if step in seen_steps:
                continue
            seen_steps.add(step)

            loss = row.get(LOSS_KEY)
            func_loss = row.get(FUNCTIONAL_LOSS_KEY)
            nonfunc_loss = row.get(NONFUNCTIONAL_LOSS_KEY)

            record: dict[str, str | float | int] = {
                "dataset": dataset,
                "model size": model_size,
                "step": int(step),
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
    """Fit a sigmoid to x vs y using robust regression."""
    k0 = 5.0 if np.corrcoef(x, y)[0, 1] >= 0 else -5.0
    p0 = np.array([y.min(), y.max(), k0, float(np.median(x))])
    bounds = ([0, 0, -np.inf, -np.inf], [1, 1, np.inf, np.inf])

    def residuals(params: np.ndarray) -> np.ndarray:
        return _sigmoid(x, *params) - y

    result = optimize.least_squares(residuals, p0, bounds=bounds, loss="soft_l1", max_nfev=10000)
    return result.x


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute coefficient of determination (R²)."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    return 1 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _correlation_subtitle(df: pd.DataFrame, xcol: str, *, r2: float | None = None) -> str:
    """Return a compact correlation string with one-sided p-values (positive direction)."""
    if len(df) < 3:
        return ""
    pearson_r, pearson_p_two = stats.pearsonr(df[xcol], df["auprc"])
    spearman_r, spearman_p_two = stats.spearmanr(df[xcol], df["auprc"])
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


def _annotate_correlation(
    ax: plt.Axes, df: pd.DataFrame, xcol: str, *, r2: float | None = None
) -> None:
    """Add correlation annotation in the emptiest corner of the axes."""
    text = _correlation_subtitle(df, xcol, r2=r2)
    if not text:
        return
    x, y, ha, va = _emptiest_corner(ax)
    ax.annotate(text, xy=(x, y), xycoords="axes fraction", fontsize=9, ha=ha, va=va)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """1×4 scatter: one panel per LL metric, shared y-axis (AUPRC)."""
    fig, axes = plt.subplots(1, len(XCOLS), sharey=True, figsize=(SUBPLOT_SIZE * len(XCOLS), SUBPLOT_SIZE))

    vmin = df["step"].min()
    vmax = df["step"].max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    for ax, xcol in zip(axes, XCOLS):
        panel_df = df.dropna(subset=[xcol])

        for _, row in panel_df.iterrows():
            ax.scatter(
                row[xcol],
                row["auprc"],
                c=[cmap(norm(row["step"]))],
                marker=DATASET_MARKERS[row["dataset"]],
                s=MODEL_SIZE_POINTS[row["model size"]],
                edgecolors="none",
            )

        # Sigmoid fit
        r2 = None
        if len(panel_df) >= 4:
            x = panel_df[xcol].values
            y = panel_df["auprc"].values
            try:
                popt = _fit_sigmoid(x, y)
                r2 = _r_squared(y, _sigmoid(x, *popt))
                x_fit = np.linspace(x.min(), x.max(), 200)
                ax.plot(x_fit, _sigmoid(x_fit, *popt), color="C3", linewidth=1.5)
            except Exception:
                logger.warning("Sigmoid fit failed for %s", xcol)

        _annotate_correlation(ax, panel_df, xcol, r2=r2)
        ax.set_xlabel(xcol)

    axes[0].set_ylabel("Promoter VEP AUPRC")
    sns.despine(fig=fig)

    # Shared colorbar for step
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), label="Step", shrink=0.8, pad=0.02)

    # Proxy-artist legends
    dataset_handles = [
        plt.Line2D(
            [], [],
            marker=DATASET_MARKERS[d],
            color="none",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=7,
            label=d,
        )
        for d in DATASET_ORDER
    ]
    size_handles = [
        plt.Line2D(
            [], [],
            marker="o",
            color="none",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=np.sqrt(MODEL_SIZE_POINTS[s]),
            label=s,
        )
        for s in MODEL_SIZE_ORDER
    ]
    fig.legend(
        handles=dataset_handles + size_handles,
        loc="center right",
        fontsize=8,
        frameon=False,
        ncol=1,
        bbox_to_anchor=(1.14, 0.5),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


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

    plot_scatter(df, OUTPUT_DIR / "functional_pos_scatter.svg")


if __name__ == "__main__":
    main()
