"""Training curves of mendelian PairwiseAccuracy for exp187 per-region arms.

Plots one panel per mendelian subset (8 real + `_global_` + `_macro_avg_`) with
one line per training-region arm (6 arms), x = training step, y = PA on
`minus_llr` (the headline score_type for the mendelian leaderboard).

The expected diagonal-winner per subset (from the plan's hypothesis table) is
highlighted with a thicker line so visual scanning matches the plan:

    | training region          | wins on mendelian subset(s)         |
    | v3_cds                   | missense, synonymous, splicing      |
    | v3_utr3                  | 3' UTR                              |
    | v3_ncrna_exon            | ncRNA                               |
    | v3_tss_region_and_utr5   | 5' UTR, tss_proximal                |
    | v3_ccre_non_promoter     | distal                              |
    | v3_bg                    | (none — control)                    |

Reads metric parquets directly from S3 via polars; no local copies needed.
Outputs land under `plots/output/` (gitignored).

Run:
    uv run python plots/exp187_per_region.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# 6 training-region arms in plan order. Names match the `name:` keys in
# snakemake/analysis/evals_v2/config/config.yaml entries for exp187.
ARMS: list[str] = [
    "v3_cds",
    "v3_utr3",
    "v3_ncrna_exon",
    "v3_tss_region_and_utr5",
    "v3_ccre_non_promoter",
    "v3_bg",
]

# 5 HF-export steps per arm — 1k/2k/3k/4k are the new (this PR) evals,
# 4999 is the final post-training checkpoint already computed.
STEPS: list[int] = [1000, 2000, 3000, 4000, 4999]

# Headline score_type for mendelian PA — per PR #186, `minus_llr` is the
# default scoring direction (pathogenic > benign) for this dataset.
SCORE_TYPE: str = "minus_llr"

S3_PREFIX: str = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"

# Diagonal-wins map from the plan (training region → subsets it should win).
# Used only for visual highlighting (thicker line on the matching panel).
DIAGONAL: dict[str, set[str]] = {
    "v3_cds": {"missense_variant", "synonymous_variant", "splicing"},
    "v3_utr3": {"3_prime_UTR_variant"},
    "v3_ncrna_exon": {"non_coding_transcript_exon_variant"},
    "v3_tss_region_and_utr5": {"5_prime_UTR_variant", "tss_proximal"},
    "v3_ccre_non_promoter": {"distal"},
    "v3_bg": set(),
}

# 6 colorblind-friendly colors (Okabe-Ito) — one per arm.
ARM_COLORS: dict[str, str] = {
    "v3_cds": "#E69F00",                  # orange
    "v3_utr3": "#56B4E9",                 # sky blue
    "v3_ncrna_exon": "#009E73",           # bluish green
    "v3_tss_region_and_utr5": "#F0E442",  # yellow
    "v3_ccre_non_promoter": "#0072B2",    # blue
    "v3_bg": "#D55E00",                   # vermillion
}


def load_all() -> pl.DataFrame:
    """Load all 30 (arm × step) metric parquets and concatenate."""
    frames: list[pl.DataFrame] = []
    for arm in ARMS:
        for step in STEPS:
            uri = f"{S3_PREFIX}/exp187-{arm}-step-{step}/mendelian_traits.parquet"
            df = pl.read_parquet(uri).with_columns(
                pl.lit(arm).alias("arm"),
                pl.lit(step).alias("step"),
            )
            frames.append(df)
    out = pl.concat(frames, how="vertical")
    # Defensive: confirm shape — 30 files × 20 rows (10 subsets × 2 score_types).
    assert out.height == 30 * 20, f"unexpected row count {out.height}"
    return out


def main() -> None:
    print("Loading 30 parquets from S3 ...")
    df = load_all().filter(pl.col("score_type") == SCORE_TYPE)
    print(f"  rows after score_type filter: {df.height}")

    subsets = sorted(df["subset"].unique().to_list())
    print(f"  subsets ({len(subsets)}): {subsets}")

    # Panel order: real subsets first (alphabetical), then the two sentinels.
    sentinels = ["_global_", "_macro_avg_"]
    real = [s for s in subsets if s not in sentinels]
    panel_order = sorted(real) + [s for s in sentinels if s in subsets]

    n_panels = len(panel_order)
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )

    for i, subset in enumerate(panel_order):
        ax = axes[i // n_cols][i % n_cols]
        sub = df.filter(pl.col("subset") == subset).sort(["arm", "step"])
        for arm in ARMS:
            arm_df = sub.filter(pl.col("arm") == arm).sort("step")
            if arm_df.is_empty():
                continue
            xs = arm_df["step"].to_list()
            ys = arm_df["value"].to_list()
            es = arm_df["se"].to_list()
            highlight = subset in DIAGONAL[arm]
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                marker="o",
                markersize=5 if highlight else 4,
                linewidth=2.5 if highlight else 1.2,
                color=ARM_COLORS[arm],
                label=arm,
                capsize=2,
                alpha=0.95 if highlight else 0.75,
            )
        ax.axhline(0.5, linestyle=":", color="gray", linewidth=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel("PairwiseAccuracy")
        title = subset
        # Mark sentinel rows so the eye knows they're aggregates.
        if subset == "_global_":
            title = "_global_ (variant-weighted avg)"
        elif subset == "_macro_avg_":
            title = "_macro_avg_ (subset-weighted avg) [HEADLINE]"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide any unused panels.
    for j in range(n_panels, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    # Single shared legend below the grid.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(ARMS),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )

    fig.suptitle(
        f"exp187 — Mendelian PA training curves ({SCORE_TYPE})\n"
        "thick line = expected diagonal winner per plan; dotted = chance",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "exp187_pa_training_curves.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"Wrote {out_png}")

    # Also dump the macro-avg-only summary as a quick text table.
    macro = (
        df.filter(pl.col("subset") == "_macro_avg_")
        .select(["arm", "step", "value", "se"])
        .sort(["arm", "step"])
    )
    print("\n_macro_avg_ PA (headline) by arm × step:")
    print(macro)


if __name__ == "__main__":
    main()
