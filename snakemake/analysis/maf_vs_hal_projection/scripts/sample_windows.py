"""Sample benchmark windows from the upstream conservation-filtered BED.

Reads ``min0.20.bed.gz`` (output of ``snakemake/zoonomia_projection_dataset``,
Ensembl chrom names ``1, 2, ...``), restricts to ``chr1`` (``1`` in the input),
samples ``--n`` rows at ``--seed``, normalizes chrom names to UCSC ``chr1``
form (matching the Cactus alignment's anchor row format), and writes a 4-col
BED for halLiftover and the MAF stream to consume.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import polars as pl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="min0.20.bed.gz")
    p.add_argument("--chrom", default="1", help="Ensembl chrom to subset to")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True, help="output BED4")
    args = p.parse_args()

    df = pl.read_csv(
        args.input,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name"],
        schema_overrides={
            "chrom": pl.Utf8,
            "start": pl.Int64,
            "end": pl.Int64,
            "name": pl.Utf8,
        },
    )
    chr_mask = df["chrom"] == args.chrom
    df = df.filter(chr_mask)
    assert df.height > 0, f"no rows with chrom={args.chrom!r} in {args.input}"

    # Sample with a fixed seed.
    sampled = df.sample(n=min(args.n, df.height), seed=args.seed).sort(
        ["chrom", "start"]
    )
    assert (sampled["end"] - sampled["start"] == 255).all()

    # Normalize Ensembl '1' -> UCSC 'chr1' to match the MAF anchor row chrom
    # (`Homo_sapiens.chr1`) and halLiftover's expected source chrom. (Cactus
    # alignments use UCSC chrom names internally.)
    out = sampled.with_columns(
        pl.col("chrom").map_elements(
            lambda c: c if c.startswith("chr") else f"chr{c}", return_dtype=pl.Utf8
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(args.output).endswith(".gz") else open
    with opener(args.output, "wt") as fout:
        for row in out.iter_rows(named=True):
            fout.write(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['name']}\n")
    print(f"wrote {out.height} windows to {args.output}")


if __name__ == "__main__":
    main()
