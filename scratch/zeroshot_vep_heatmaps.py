"""Render heatmaps + top-N score rankings from the zeroshot_vep aggregated parquet.

Usage:
    uv run python scratch/zeroshot_vep_heatmaps.py s3://oa-bolinas/snakemake/analysis/zeroshot_vep/results/metrics_aggregated.parquet --out-dir scratch/zeroshot_plots/

Renders three plot families per dataset:
1. `global_pooled` PairwiseAccuracy: score × (model, window) heatmap.
2. `global_macro` PairwiseAccuracy: same axes, different aggregation.
3. Per-subset detail: score × subset heatmap, one per (model, window).

Plus a flat top-N table of (score, model, window) ranked by pooled accuracy per
dataset, for the issue-body summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Defer matplotlib import so this script is cheap to run for the table-only
# code path on the local node.
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


DATASETS = ("mendelian_traits", "complex_traits", "eqtl")


def _heatmap(
    pivoted: pd.DataFrame,
    title: str,
    out_path: Path,
    vmin: float = 0.4,
    vmax: float = 0.85,
    cmap: str = "RdBu_r",
    fmt: str = ".3f",
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * pivoted.shape[1]),
                                    max(6, 0.3 * pivoted.shape[0])))
    im = ax.imshow(pivoted.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(pivoted.shape[1]))
    ax.set_xticklabels(pivoted.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivoted.shape[0]))
    ax.set_yticklabels(pivoted.index, fontsize=8)
    for i in range(pivoted.shape[0]):
        for j in range(pivoted.shape[1]):
            val = pivoted.values[i, j]
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=6, color="black" if 0.45 < val < 0.65 else "white")
    plt.colorbar(im, ax=ax, label="PairwiseAccuracy")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def render(agg: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg["mw"] = agg["model"] + " | w" + agg["window"].astype(str)

    for dataset in DATASETS:
        sub_ds = agg[agg["dataset"] == dataset]
        if len(sub_ds) == 0:
            print(f"[heatmaps] no rows for dataset={dataset}; skip", file=sys.stderr)
            continue

        for agg_name in ("global_pooled", "global_macro"):
            df = sub_ds[sub_ds["aggregation"] == agg_name]
            if len(df) == 0:
                continue
            pivot = df.pivot_table(
                index="score", columns="mw", values="value", aggfunc="first"
            )
            if HAS_PLT:
                _heatmap(
                    pivot, f"{dataset} — {agg_name}",
                    out_dir / f"{dataset}_{agg_name}.png",
                )

        # Per-subset detail: score × subset, best (model, window) per cell.
        per_subset = sub_ds[sub_ds["aggregation"] == "per_subset"]
        if len(per_subset) > 0:
            # Aggregate across (model, window) — take max for "best score on this subset".
            best = per_subset.groupby(["score", "subset"])["value"].max().reset_index()
            pivot_best = best.pivot_table(
                index="score", columns="subset", values="value"
            )
            if HAS_PLT:
                _heatmap(
                    pivot_best,
                    f"{dataset} — per-subset (best across model × window)",
                    out_dir / f"{dataset}_per_subset_best.png",
                )

    # Top-10 ranking per dataset (pooled).
    summaries = []
    for dataset in DATASETS:
        pool = agg[(agg["dataset"] == dataset) & (agg["aggregation"] == "global_pooled")]
        top = pool.sort_values("value", ascending=False).head(10)[
            ["model", "window", "score", "value", "se", "n_pairs"]
        ].copy()
        top["dataset"] = dataset
        summaries.append(top)
    if summaries:
        top_df = pd.concat(summaries, ignore_index=True)
        top_df.to_csv(out_dir / "top10_per_dataset.csv", index=False)
        print(top_df.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", help="Path or s3:// URI to metrics_aggregated.parquet")
    ap.add_argument("--out-dir", default="scratch/zeroshot_plots")
    args = ap.parse_args(argv)

    agg = pd.read_parquet(args.parquet)
    print(f"[heatmaps] loaded {len(agg)} rows across "
          f"{agg.groupby(['model', 'window', 'dataset']).ngroups} (model, window, dataset) combos")
    render(agg, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
