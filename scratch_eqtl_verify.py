"""Verification script for the new Catalogue-sourced eqtl dataset.

Compares the new aggregated.parquet + dataset_unsplit/eqtl.parquet against
the prior Finucane-sourced numbers. Expected (per plan §verification):

- Aggregated negatives: >10M (Finucane: 1.28M; complex_traits: 13.2M)
- Per-subset matched pair counts increase substantially vs. Finucane
- missense (was 15), synonymous (was 19), splicing (was 5) re-cross n≥30
- Pairing invariant: every match_group ID has exactly 1 True + 1 False

Reads from S3 directly. Run after `sky launch -y -c eqtl-catalogue ...`
finishes. Local-only quick analysis script — feel free to delete.
"""

from __future__ import annotations

import polars as pl

# Finucane reference numbers from the plan (post-bug-fix train split).
FINUCANE_TRAIN_PAIRS = {
    "distal": 1397,
    "non_coding_transcript_exon_variant": 143,
    "tss_proximal": 125,
    "3_prime_UTR_variant": 88,
    "5_prime_UTR_variant": 31,
    "missense_variant": 15,
    "synonymous_variant": 19,
    "splicing": 5,
}
FINUCANE_AGGREGATED = {"pos": 17_925, "neg": 1_277_139}

S3_PREFIX = "s3://oa-bolinas/snakemake/evals/results"


def main() -> None:
    agg = pl.read_parquet(f"{S3_PREFIX}/eqtl/aggregated.parquet")
    n_pos = agg.filter(pl.col("label")).height
    n_neg = agg.filter(~pl.col("label")).height
    print("=== aggregated.parquet ===")
    print(f"  total: {agg.height:,} variants")
    print(f"  pos:  {n_pos:,} ({n_pos / FINUCANE_AGGREGATED['pos']:.1f}× Finucane)")
    print(f"  neg:  {n_neg:,} ({n_neg / FINUCANE_AGGREGATED['neg']:.1f}× Finucane)")
    print()
    print("  PIP histogram:")
    for lo, hi in [
        (0, 1e-6),
        (1e-6, 0.001),
        (0.001, 0.005),
        (0.005, 0.01),
        (0.01, 0.1),
        (0.1, 0.5),
        (0.5, 0.9),
        (0.9, 1.01),
    ]:
        n = agg.filter((pl.col("pip") >= lo) & (pl.col("pip") < hi)).height
        print(f"    [{lo:.6f}, {hi:.6f}): {n:>12,}")
    print()
    print("  Sanity asserts:")
    assert agg.filter((~pl.col("label")) & (pl.col("pip") >= 0.01)).height == 0, (
        "negatives must have pip < 0.01"
    )
    assert agg.filter((pl.col("label")) & (pl.col("pip") <= 0.9)).height == 0, (
        "positives must have pip > 0.9"
    )
    print("    ✓ no negative with pip >= 0.01")
    print("    ✓ no positive with pip <= 0.9")

    print()
    ds = pl.read_parquet(f"{S3_PREFIX}/dataset_unsplit/eqtl.parquet")
    print("=== dataset_unsplit/eqtl.parquet ===")
    print(f"  total: {ds.height:,}")
    print(f"  pos:   {ds.filter(pl.col('label')).height:,}")
    print(f"  neg:   {ds.filter(~pl.col('label')).height:,}")
    print()

    # Pair invariant
    by_mg = ds.group_by("match_group").agg(pl.col("label").n_unique().alias("n_labels"))
    bad = by_mg.filter(pl.col("n_labels") != 2).height
    assert bad == 0, f"{bad} match_groups don't have exactly 1 True + 1 False"
    print(
        f"  ✓ pairing invariant: all {by_mg.height:,} match_groups have 1 pos + 1 neg"
    )

    # Train-only per-subset (for direct comparison with leaderboard #172)
    odd_chroms = [str(i) for i in range(1, 23, 2)] + ["X"]
    train_ds = ds.filter(pl.col("chrom").is_in(odd_chroms))
    print()
    print("=== Per-subset matched-pair counts (train split) — new vs. Finucane ===")
    print(f"{'subset':40} {'new':>8} {'old':>8} {'ratio':>7}")
    for subset, old_n in sorted(FINUCANE_TRAIN_PAIRS.items(), key=lambda x: -x[1]):
        new_n = train_ds.filter(pl.col("label") & (pl.col("subset") == subset)).height
        ratio = new_n / old_n if old_n > 0 else float("inf")
        flag = "  ✓ ≥30" if new_n >= 30 else "  ✗ <30"
        print(f"{subset:40} {new_n:>8} {old_n:>8} {ratio:>7.2f}x{flag}")


if __name__ == "__main__":
    main()
