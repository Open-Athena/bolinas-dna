"""One-off: per-stratum (chrom × subset × candidate-bin) pos/neg counts.

Used to inform bin-edge choices for the four stubborn AUPRC-leak cells in
mendelian:
  - tss_proximal × distance_tss_pc
  - tss_proximal × distance_exon_pc
  - splicing      × distance_exon_pc
  - distal        × distance_exon_pc

For each candidate bin scheme, reports per-stratum neg counts so we can
predict subsampling drops at k=9 before committing.
"""

from __future__ import annotations

import polars as pl

DATASET_ALL = (
    "s3://oa-bolinas/snakemake/evals/results/mendelian_traits/dataset_all.parquet"
)
OUT = "s3://oa-bolinas/snakemake/evals/results/scratch/bin_distribution_check.parquet"

CELLS = [
    ("tss_proximal", "distance_tss_pc"),
    ("tss_proximal", "distance_exon_pc"),
    ("splicing", "distance_exon_pc"),
    ("distal", "distance_exon_pc"),
]

K = 9  # matching ratio


def quantile_row(df: pl.DataFrame, col: str, q: float) -> float | None:
    return df[col].quantile(q) if df.height else None


def main() -> None:
    print(f"Reading {DATASET_ALL} …", flush=True)
    da = pl.read_parquet(DATASET_ALL)
    print(
        f"  rows={da.height:,}  pos={da.filter(pl.col('label')).height:,}  "
        f"neg={da.filter(~pl.col('label')).height:,}",
        flush=True,
    )

    rows: list[dict] = []
    for subset, feat in CELLS:
        sub = da.filter(pl.col("consequence_group") == subset)
        pos = sub.filter(pl.col("label"))
        neg = sub.filter(~pl.col("label"))
        n_pos, n_neg = pos.height, neg.height
        print(
            f"\n=== {subset} × {feat}  (pos={n_pos:,}, neg={n_neg:,}) ===", flush=True
        )
        print("  quantile      pos        neg", flush=True)
        for q in [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]:
            p = quantile_row(pos, feat, q)
            n = quantile_row(neg, feat, q)
            ps = f"{p:>10.0f}" if p is not None else " " * 10
            ns = f"{n:>10.0f}" if n is not None else " " * 10
            print(f"     q{int(q * 100):02d}   {ps}   {ns}", flush=True)
            rows.append(
                {
                    "subset": subset,
                    "feature": feat,
                    "quantile": q,
                    "pos_val": p,
                    "neg_val": n,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }
            )

    out = pl.DataFrame(rows)
    print(f"\nWriting summary to {OUT}", flush=True)
    out.write_parquet(OUT)


if __name__ == "__main__":
    main()
