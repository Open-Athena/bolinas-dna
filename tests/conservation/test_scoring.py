"""Tests for ``bolinas.conservation.scoring``."""

from __future__ import annotations

import polars as pl

from bolinas.conservation.scoring import parse_bigwig_average_over_bed


def test_parse_bigwig_average_over_bed_dtypes(tmp_path) -> None:
    """Schema, dtypes, and columns match the bigWigAverageOverBed contract."""
    p = tmp_path / "out.tsv"
    p.write_text(
        "win_001\t255\t255\t100.5\t0.394\t0.394\n"
        "win_002\t255\t250\t40.0\t0.157\t0.160\n"
    )
    df = parse_bigwig_average_over_bed(p)
    assert df.columns == ["name", "size", "covered", "sum", "mean0", "mean"]
    assert df.schema["name"] == pl.Utf8
    assert df.schema["size"] == pl.Int64
    assert df.schema["covered"] == pl.Int64
    assert df.schema["sum"] == pl.Float64
    assert df.schema["mean0"] == pl.Float64
    assert df.schema["mean"] == pl.Float64
    assert df.shape == (2, 6)


def test_parse_bigwig_average_over_bed_handles_zero_covered(tmp_path) -> None:
    """When ``covered == 0``, ``mean`` is ``n/a`` in the TSV; we coerce to NaN."""
    p = tmp_path / "out.tsv"
    p.write_text("win_001\t255\t0\t0.0\t0.0\tn/a\n")
    df = parse_bigwig_average_over_bed(p)
    assert df["covered"][0] == 0
    assert df["sum"][0] == 0.0
    assert df["mean0"][0] == 0.0
    assert df["mean"].is_null().all()


def test_parse_bigwig_average_over_bed_binary_track(tmp_path) -> None:
    """For a binarized bigWig, sum == conserved_bases and mean0 == proportion.
    Smoke test of the math we depend on downstream."""
    p = tmp_path / "out.tsv"
    p.write_text(
        # window of size 255, all 255 covered, 80 conserved bases
        "win_001\t255\t255\t80\t0.31373\t0.31373\n"
    )
    df = parse_bigwig_average_over_bed(p)
    assert df["sum"][0] == 80.0
    assert abs(df["mean0"][0] - 80 / 255) < 1e-3
