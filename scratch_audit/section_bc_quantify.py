"""Section B + C — dataset-wide N and lowercase quantification.

Run on the SkyPilot cluster. Assumes:
  - ~/audit/all_species_with_sequence.parquet
  - ~/audit/species_zoonomia_447_family_dedup.tsv

Emits to ~/audit/section_bc/:
  - per_species_stats.tsv            — per-species N + lowercase stats, joined with metadata
  - per_species_stats.parquet
  - n_position_histogram.tsv
  - by_accession_prefix.tsv
  - by_quality_source.tsv
  - by_assembly_level.tsv

Strategy: do all aggregation through Polars streaming (no quantile aggs inside
the group_by — those don't stream well in 1.x). Per-species lowercase quantiles
are computed in a separate pass via pre-aggregation if needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


WD = Path.home() / "audit"
OUT = WD / "section_bc"
OUT.mkdir(parents=True, exist_ok=True)

SRC = WD / "all_species_with_sequence.parquet"
META_TSV = WD / "species_zoonomia_447_family_dedup.tsv"


def add_seq_stats(lf: pl.LazyFrame) -> pl.LazyFrame:
    s = pl.col("sequence")
    return lf.with_columns([
        s.str.len_bytes().alias("seq_len"),
        (s.str.count_matches("N", literal=True)
         + s.str.count_matches("n", literal=True)).alias("n_count"),
        (s.str.count_matches("a", literal=True)
         + s.str.count_matches("c", literal=True)
         + s.str.count_matches("g", literal=True)
         + s.str.count_matches("t", literal=True)).alias("lower_count"),
        # Position of first / last N within the window:
        s.str.find("N").alias("first_n_idx"),
        s.str.reverse().str.find("N").alias("last_n_from_end"),
    ]).with_columns([
        (pl.col("seq_len") - pl.col("n_count")).alias("non_n_len"),
        (pl.col("n_count") / pl.col("seq_len")).alias("n_frac"),
    ])


def main():
    if not SRC.exists():
        sys.exit(f"missing source parquet: {SRC}")
    if not META_TSV.exists():
        sys.exit(f"missing species metadata TSV: {META_TSV}")

    print(f"[BC] Scanning {SRC} ({SRC.stat().st_size / 1e9:.2f} GB)...", flush=True)

    meta = pl.read_csv(META_TSV, separator="\t").with_columns(
        accession_prefix=pl.col("accession").str.slice(0, 3)  # 'GCF' or 'GCA'
    )

    lf = pl.scan_parquet(str(SRC))
    lf = add_seq_stats(lf)

    EDGE = 32  # bp from either end counts as 'leading' / 'trailing'

    # N-position bucket categorical (computed lazily, no Python loop).
    bucket = (
        pl.when(pl.col("n_count") == 0).then(pl.lit("none"))
        .when((pl.col("first_n_idx") < EDGE) & (pl.col("last_n_from_end") >= EDGE)).then(pl.lit("leading"))
        .when((pl.col("first_n_idx") >= EDGE) & (pl.col("last_n_from_end") < EDGE)).then(pl.lit("trailing"))
        .when((pl.col("first_n_idx") < EDGE) & (pl.col("last_n_from_end") < EDGE)).then(pl.lit("both_edges"))
        .otherwise(pl.lit("interior"))
        .alias("n_pos_bucket")
    )
    lf = lf.with_columns(bucket)

    # Aggregate per species — streaming-friendly: sums and means only.
    agg = lf.group_by("species").agg([
        pl.len().alias("n_windows"),
        pl.col("n_count").sum().alias("total_n_bases"),
        pl.col("seq_len").sum().alias("total_bases"),
        pl.col("lower_count").sum().alias("total_lower_bases"),
        pl.col("non_n_len").sum().alias("total_non_n_bases"),
        # Per-row threshold counters
        (pl.col("n_count") > 0).sum().alias("rows_any_n"),
        (pl.col("n_frac") >= 0.10).sum().alias("rows_n_ge_10pct"),
        (pl.col("n_frac") >= 0.50).sum().alias("rows_n_ge_50pct"),
        # N-position bucket counts
        (pl.col("n_pos_bucket") == "leading").sum().alias("rows_leading"),
        (pl.col("n_pos_bucket") == "trailing").sum().alias("rows_trailing"),
        (pl.col("n_pos_bucket") == "both_edges").sum().alias("rows_both_edges"),
        (pl.col("n_pos_bucket") == "interior").sum().alias("rows_interior"),
        (pl.col("n_pos_bucket") == "none").sum().alias("rows_none"),
    ])

    print("[BC] Collecting per-species aggregation (engine=streaming)...", flush=True)
    per_species = agg.collect(engine="streaming")
    print(f"[BC] per_species shape: {per_species.shape}", flush=True)

    # Per-species ratios from sums (numerically stable, no quantiles)
    per_species = per_species.with_columns([
        (pl.col("total_n_bases") / pl.col("total_bases")).alias("agg_n_frac"),
        (pl.col("total_lower_bases") / pl.col("total_non_n_bases")).alias("agg_lower_frac"),
        (pl.col("rows_any_n") / pl.col("n_windows")).alias("frac_rows_any_n"),
        (pl.col("rows_n_ge_10pct") / pl.col("n_windows")).alias("frac_rows_n_ge_10pct"),
        (pl.col("rows_n_ge_50pct") / pl.col("n_windows")).alias("frac_rows_n_ge_50pct"),
    ])

    joined = per_species.join(meta, on="species", how="left").sort("agg_n_frac", descending=True)
    joined.write_csv(OUT / "per_species_stats.tsv", separator="\t")
    joined.write_parquet(OUT / "per_species_stats.parquet")
    print(f"[BC] Wrote {OUT/'per_species_stats.tsv'} ({joined.height} rows)")

    # Position histogram across all species
    pos_hist = pl.DataFrame({
        "bucket": ["leading", "trailing", "both_edges", "interior", "none"],
        "n_rows": [
            int(joined["rows_leading"].sum()),
            int(joined["rows_trailing"].sum()),
            int(joined["rows_both_edges"].sum()),
            int(joined["rows_interior"].sum()),
            int(joined["rows_none"].sum()),
        ],
    })
    total = int(pos_hist["n_rows"].sum())
    pos_hist = pos_hist.with_columns(frac=pl.col("n_rows") / total)
    pos_hist.write_csv(OUT / "n_position_histogram.tsv", separator="\t")
    print("[BC] N-position histogram (across all v1 rows):")
    print(pos_hist)

    # Worst and best 5 by agg_n_frac
    cols = ["species", "accession", "accession_prefix", "assembly_level",
            "contig_n50", "quality_source", "n_windows",
            "agg_n_frac", "frac_rows_n_ge_10pct", "frac_rows_n_ge_50pct",
            "agg_lower_frac"]
    print("\n[BC] WORST 5 (highest agg_n_frac):")
    print(joined.head(5).select(cols))
    print("\n[BC] BEST 5 (lowest agg_n_frac):")
    print(joined.tail(5).select(cols))

    # Marginal summaries by metadata axis
    for axis in ("accession_prefix", "quality_source", "assembly_level"):
        marg = joined.group_by(axis).agg([
            pl.len().alias("n_species"),
            pl.col("agg_n_frac").mean().alias("mean_agg_n_frac"),
            pl.col("agg_lower_frac").mean().alias("mean_agg_lower_frac"),
            pl.col("frac_rows_n_ge_10pct").mean().alias("mean_frac_rows_n_ge_10pct"),
        ])
        marg.write_csv(OUT / f"by_{axis}.tsv", separator="\t")
        print(f"\n[BC] By {axis}:")
        print(marg)

    print(f"\n[BC] DONE. Outputs in {OUT}.")


if __name__ == "__main__":
    main()
