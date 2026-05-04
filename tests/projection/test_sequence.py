"""Tests for ``bolinas.projection.sequence``."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from bolinas.projection.sequence import parquet_to_bed6, revcomp


def test_revcomp_basic() -> None:
    assert revcomp("ACGT") == "ACGT"
    assert revcomp("AAAA") == "TTTT"
    assert revcomp("GGCC") == "GGCC"
    assert revcomp("ACGTN") == "NACGT"


def test_revcomp_preserves_case() -> None:
    assert revcomp("ACGTacgt") == "acgtACGT"
    assert revcomp("Nn") == "nN"


def test_revcomp_iupac_passthrough() -> None:
    """Non-ACGTN characters are returned unchanged."""
    assert revcomp("ACGTW") == "WACGT"


def _make_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    schema = {
        "query_name": pl.Utf8,
        "species": pl.Utf8,
        "t_chrom": pl.Utf8,
        "t_start": pl.Int64,
        "t_end": pl.Int64,
        "t_strand": pl.Utf8,
        "t_src_size": pl.Int64,
    }
    pq = tmp_path / "proj.parquet"
    pl.DataFrame(rows, schema=schema).write_parquet(pq)
    return pq


def test_parquet_to_bed6_writes_six_cols(tmp_path: Path) -> None:
    pq = _make_parquet(
        tmp_path,
        [
            {
                "query_name": "win_1_000000001",
                "species": "Mus_musculus",
                "t_chrom": "chr5",
                "t_start": 100,
                "t_end": 355,
                "t_strand": "+",
                "t_src_size": 1_000_000,
            },
            {
                "query_name": "win_1_000000002",
                "species": "Mus_musculus",
                "t_chrom": "chr5",
                "t_start": 500,
                "t_end": 755,
                "t_strand": "-",
                "t_src_size": 1_000_000,
            },
        ],
    )
    bed = tmp_path / "out.bed"
    n = parquet_to_bed6(pq, bed)
    assert n == 2
    assert bed.read_text() == (
        "chr5\t100\t355\twin_1_000000001\t0\t+\nchr5\t500\t755\twin_1_000000002\t0\t-\n"
    )


def test_parquet_to_bed6_empty(tmp_path: Path) -> None:
    pq = _make_parquet(tmp_path, [])
    bed = tmp_path / "out.bed"
    n = parquet_to_bed6(pq, bed)
    assert n == 0
    assert bed.exists()
    assert bed.read_text() == ""
