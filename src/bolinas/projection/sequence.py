"""Per-species sequence extraction at projected coordinates.

The Snakemake rule converts each per-species projection Parquet into a
6-column BED, then runs ``bedtools getfasta -s`` against the species
FASTA (output of ``hal2fasta``) to extract strand-aware sequences. The
helper here turns a Parquet into a BED — small glue, but tested.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


_REVCOMP_TABLE = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(seq: str) -> str:
    """Return the reverse-complement of a DNA sequence.

    Preserves case; ``N``/``n`` map to themselves. Non-ACGTN characters
    pass through unchanged (extraction from a 2bit / FASTA produces
    only ACGTN, so the pass-through behaviour only matters in tests).
    Used as a fallback when a tool that doesn't honor strand is the
    only option; the production rule uses ``bedtools getfasta -s``,
    which revcomps natively.
    """
    return seq.translate(_REVCOMP_TABLE)[::-1]


def parquet_to_bed6(parquet_path: str | Path, out_bed: str | Path) -> int:
    """Materialize a per-species projection Parquet as a BED6 file.

    Columns: ``t_chrom\\tt_start\\tt_end\\tquery_name\\t0\\tt_strand``.
    No header. The score column is always ``0`` (the projection has no
    score concept — the BED6 column is required so ``bedtools getfasta -s``
    sees a strand). Returns the number of rows written.

    Empty Parquet → empty BED, returns 0.
    """
    df = pl.read_parquet(parquet_path)
    out = Path(out_bed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        if df.is_empty():
            return 0
        for row in df.iter_rows(named=True):
            f.write(
                f"{row['t_chrom']}\t{row['t_start']}\t{row['t_end']}"
                f"\t{row['query_name']}\t0\t{row['t_strand']}\n"
            )
    return df.height
