"""RefSeq baseline — per-genome lowercase fraction in genomes-v5 mammals v1.

Streams a comparable RefSeq-pipeline HF dataset (`genomes-v5-genome_set-mammals-
intervals-v1_255_128`), computes per-genome and overall lowercase fraction so
we can compare against zoonomia-v1-v1 fairly.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import polars as pl


WD = Path.home() / "audit"
OUT = WD / "section_bc"
OUT.mkdir(parents=True, exist_ok=True)

DATASET = "bolinas-dna/genomes-v5-genome_set-mammals-intervals-v1_255_128"
SAMPLE_ROWS = 200_000  # ~5 GB rows total at 255bp; 200k subsample is plenty for distribution.


def main():
    from datasets import load_dataset

    print(f"[REFSEQ] Loading {DATASET} (streaming)...")
    ds = load_dataset(DATASET, split="train", streaming=True)

    rows = []
    n_seen = 0
    for row in ds:
        n_seen += 1
        seq = row["seq"]
        seq_len = len(seq)
        # Strip case to count nucleotide categories. Genomes-v5 sequences
        # may include N's; treat them like zoonomia.
        n_count = seq.count("N") + seq.count("n")
        lower_count = sum(1 for c in seq if c in "acgt")
        non_n_len = seq_len - n_count
        # parse genome from id: "{genome}:{chrom}:{start}-{end}" or similar
        # examples we expect to handle:
        # "NC_000067.6:0-256(+)" or simple "{chrom}:{start}-{end}"
        # genomes-v5 ids may include accession in the prefix.
        rid = row.get("id", "")
        rows.append({
            "id": rid,
            "seq_len": seq_len,
            "n_count": n_count,
            "lower_count": lower_count,
            "non_n_len": non_n_len,
        })
        if n_seen >= SAMPLE_ROWS:
            break
    print(f"[REFSEQ] sampled {len(rows)} rows.")

    df = pl.DataFrame(rows)
    # Per-row stats
    df = df.with_columns([
        (pl.col("n_count") / pl.col("seq_len")).alias("n_frac"),
        pl.when(pl.col("non_n_len") > 0).then(
            pl.col("lower_count") / pl.col("non_n_len")
        ).otherwise(0.0).alias("lower_frac"),
    ])

    # Overall summary
    total_bases = df["seq_len"].sum()
    total_n = df["n_count"].sum()
    total_lower = df["lower_count"].sum()
    total_non_n = df["non_n_len"].sum()

    summary = {
        "dataset": DATASET,
        "n_rows_sampled": df.height,
        "total_bases": total_bases,
        "total_n_bases": total_n,
        "agg_n_frac": total_n / total_bases,
        "agg_lower_frac": total_lower / max(total_non_n, 1),
        "mean_lower_frac": float(df["lower_frac"].mean()),
        "mean_n_frac": float(df["n_frac"].mean()),
    }

    print(json.dumps(summary, indent=2))
    (OUT / "refseq_baseline_summary.json").write_text(json.dumps(summary, indent=2))
    df.write_parquet(OUT / "refseq_baseline_sample.parquet")
    print(f"[REFSEQ] DONE. Wrote {OUT/'refseq_baseline_summary.json'}.")


if __name__ == "__main__":
    main()
