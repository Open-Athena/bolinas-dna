"""Plot minus_llr_avg AUPRC vs training step for the five exp55 evolutionary
timescales (humans / primates / mammals / vertebrates / animals), on two
mendelian subsets (Promoter = tss_proximal, 5' UTR = 5_prime_UTR_variant).

Mirrors the reference figure shared in chat — two panels side-by-side, one
line per arm, AUPRC on y-axis. Reads metrics parquets directly from S3.

Usage:
    uv run python scratch/plot_exp55_evolutionary_timescales.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

ARMS: tuple[str, ...] = ("humans", "primates", "mammals", "vertebrates", "animals")
ARM_LABELS: dict[str, str] = {
    "humans": "Humans (0 Mya, 1 species)",
    "primates": "Primates (~65 Mya, 11 species)",
    "mammals": "Mammals (~100 Mya, 81 species)",
    "vertebrates": "Vertebrates (~600 Mya, 317 species)",
    "animals": "Animals (~800 Mya, 499 species)",
}
# Colors picked to match the reference figure: a viridis-like sweep from
# dark purple (humans) → blue (primates) → teal (mammals) → green
# (vertebrates) → yellow (animals).
ARM_COLORS: dict[str, str] = {
    "humans": "#3b1a64",
    "primates": "#3a6fa0",
    "mammals": "#2e8c8c",
    "vertebrates": "#5bb35b",
    "animals": "#e8d840",
}
STEPS: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000, 9000, 13000, 16999)
# Panel name → subset key in the metrics parquet.
SUBSETS: dict[str, str] = {
    "Promoter": "tss_proximal",
    "5' UTR": "5_prime_UTR_variant",
}
S3_BASE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
SCORE_TYPE = "minus_llr_avg"
OUT_PATH = Path(__file__).parent / "exp55_evolutionary_timescales_auprc.png"


def load_all() -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    for arm in ARMS:
        for step in STEPS:
            uri = f"{S3_BASE}/exp55-{arm}-step-{step}/mendelian_traits.parquet"
            try:
                df = pl.read_parquet(uri)
            except Exception as exc:
                missing.append(f"  {arm}-step-{step}: {exc}")
                continue
            parts.append(
                df.with_columns(
                    pl.lit(arm).alias("arm"),
                    pl.lit(step).alias("step"),
                )
            )
    if missing:
        print(
            f"WARNING: {len(missing)} of {len(ARMS) * len(STEPS)} parquets unreadable:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, "no parquets loaded — has the sweep started?"
    return pl.concat(parts)


def main() -> None:
    all_df = load_all()
    df = all_df.filter(
        (pl.col("score_type") == SCORE_TYPE)
        & (pl.col("subset").is_in(list(SUBSETS.values())))
    )
    assert not df.is_empty(), (
        f"empty after filtering on {SCORE_TYPE} + {list(SUBSETS.values())}; "
        f"got score_types={sorted(all_df['score_type'].unique().to_list())}, "
        f"subsets={sorted(all_df['subset'].unique().to_list())}"
    )

    fig, axes = plt.subplots(1, len(SUBSETS), figsize=(11, 4), sharex=True)
    fig.suptitle("Exp55 Evolutionary Timescales — promoters, Qwen 0.6B", y=1.02)

    for ax, (panel_label, subset_key) in zip(axes, SUBSETS.items()):
        sub = df.filter(pl.col("subset") == subset_key)
        head = sub.row(0, named=True)
        n_groups = int(head["n_groups"])
        n_rows = int(head["n_rows"])
        for arm in ARMS:
            arm_df = sub.filter(pl.col("arm") == arm).sort("step")
            if arm_df.is_empty():
                continue
            ax.plot(
                arm_df["step"].to_numpy(),
                arm_df["value"].to_numpy(),
                marker="o",
                color=ARM_COLORS[arm],
                label=ARM_LABELS[arm],
                linewidth=1.5,
                markersize=5,
            )
        ax.set_title(f"{panel_label}\n(n={n_groups} vs. {n_rows - n_groups})")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("AUPRC")
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), title="Model", frameon=False
    )
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
