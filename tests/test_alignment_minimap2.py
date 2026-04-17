"""Tests for bolinas.alignment.minimap2 PAF parsing helpers."""

from __future__ import annotations

import polars as pl

from bolinas.alignment.minimap2 import PAF_SCHEMA, best_hit_per_query, parse_paf


def _paf_line(
    query: str,
    qlen: int,
    qstart: int,
    qend: int,
    strand: str,
    chrom: str,
    tlen: int,
    tstart: int,
    tend: int,
    matches: int,
    alnlen: int,
    mapq: int,
    *tags: str,
) -> str:
    fields = [
        query,
        qlen,
        qstart,
        qend,
        strand,
        chrom,
        tlen,
        tstart,
        tend,
        matches,
        alnlen,
        mapq,
        *tags,
    ]
    return "\t".join(str(f) for f in fields) + "\n"


def test_parse_paf_empty(tmp_path):
    paf = tmp_path / "empty.paf"
    paf.write_text("")
    df = parse_paf(paf)
    assert df.height == 0
    assert df.schema == PAF_SCHEMA


def test_parse_paf_ignores_sam_tags(tmp_path):
    paf = tmp_path / "with_tags.paf"
    paf.write_text(
        _paf_line(
            "q1", 100, 0, 100, "+", "chr1", 1000, 200, 300, 90, 100, 60, "tp:A:P"
        )
    )
    df = parse_paf(paf)
    assert df.height == 1
    row = df.to_dicts()[0]
    assert row["query"] == "q1"
    assert row["strand"] == "+"
    assert row["chrom"] == "chr1"
    assert row["start"] == 200
    assert row["end"] == 300
    assert row["matches"] == 90
    assert row["alnlen"] == 100
    assert row["mapq"] == 60


def test_parse_paf_skips_blank_lines(tmp_path):
    paf = tmp_path / "blanks.paf"
    paf.write_text(
        "\n"
        + _paf_line("q1", 50, 0, 50, "-", "chr2", 500, 10, 60, 40, 50, 30)
        + "\n"
    )
    df = parse_paf(paf)
    assert df.height == 1
    assert df["query"].to_list() == ["q1"]


def test_best_hit_per_query_picks_highest_matches(tmp_path):
    paf = tmp_path / "multi.paf"
    paf.write_text(
        _paf_line("q1", 100, 0, 100, "+", "chr1", 1000, 0, 100, 50, 100, 1)
        + _paf_line("q1", 100, 0, 100, "+", "chr1", 1000, 500, 600, 90, 100, 60)
        + _paf_line("q2", 80, 0, 80, "-", "chr3", 2000, 10, 90, 70, 80, 40)
    )
    df = parse_paf(paf)
    best = best_hit_per_query(df)
    assert best.height == 2
    q1 = best.filter(pl.col("query") == "q1").to_dicts()[0]
    assert q1["matches"] == 90
    assert q1["start"] == 500
    q2 = best.filter(pl.col("query") == "q2").to_dicts()[0]
    assert q2["chrom"] == "chr3"


def test_best_hit_per_query_empty():
    empty = pl.DataFrame(schema=PAF_SCHEMA)
    out = best_hit_per_query(empty)
    assert out.height == 0
    assert out.schema == PAF_SCHEMA
