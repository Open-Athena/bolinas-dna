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


def parse_bedtools_getfasta_output(fasta_path: str | Path) -> list[str]:
    """Parse a single-line-per-record FASTA from ``bedtools getfasta -nameOnly``.

    bedtools getfasta emits two lines per BED row in BED order:
    a header ``>{name}({strand})`` and a single line of sequence. We
    rely on the single-line-per-record convention (no width wrapping
    at default settings) for fast row-aligned consumption — the
    returned list of sequences is in BED row order, ready to
    ``zip(parquet_rows, sequences)``.

    Asserts every other line starts with ``>``; raises on a malformed
    file.
    """
    seqs: list[str] = []
    with Path(fasta_path).open() as f:
        for line_no, line in enumerate(f):
            stripped = line.rstrip("\n")
            if line_no % 2 == 0:
                assert stripped.startswith(">"), (
                    f"expected header at line {line_no + 1}, got: {stripped[:60]!r}"
                )
            else:
                seqs.append(stripped)
    return seqs


def attach_sequences_to_parquet(
    proj_parquet: str | Path,
    sequences: list[str],
    out_parquet: str | Path,
    *,
    target_len: int,
) -> int:
    """Append a ``sequence`` column to a per-species projection Parquet.

    ``sequences`` must align row-for-row with ``proj_parquet`` (the BED
    written by :func:`parquet_to_bed6` and consumed by
    ``bedtools getfasta`` is in the same order). Asserts every sequence
    is exactly ``target_len`` characters and that the count matches.

    Empty Parquet → empty Parquet with the extended schema.
    """
    df = pl.read_parquet(proj_parquet)
    out = Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    extended_schema = {**df.schema, "sequence": pl.Utf8}
    if df.is_empty():
        assert len(sequences) == 0
        pl.DataFrame(schema=extended_schema).write_parquet(out)
        return 0
    assert len(sequences) == df.height, (
        f"sequence count ({len(sequences)}) != Parquet rows ({df.height})"
    )
    for s in sequences:
        assert len(s) == target_len, (
            f"unexpected sequence length {len(s)}, expected {target_len}"
        )
    df_with_seq = df.with_columns(sequence=pl.Series("sequence", sequences))
    df_with_seq.write_parquet(out)
    return df.height
