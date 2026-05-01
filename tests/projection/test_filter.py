import polars as pl

from bolinas.projection.filter import filter_length, filter_single_chrom_strand


def _mkdf(rows):
    return pl.DataFrame(
        rows,
        schema={
            "query_name": pl.Utf8,
            "species": pl.Utf8,
            "t_chrom": pl.Utf8,
            "t_start": pl.Int64,
            "t_end": pl.Int64,
            "t_strand": pl.Utf8,
            "t_src_size": pl.Int64,
        },
        orient="row",
    )


def test_single_chrom_strand_merges_two_blocks():
    df = _mkdf(
        [
            ("q1", "Mus_musculus", "chr5", 1000, 1100, "+", 200_000),
            ("q1", "Mus_musculus", "chr5", 1100, 1200, "+", 200_000),
        ]
    )
    out = filter_single_chrom_strand(df)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["t_start"] == 1000
    assert row["t_end"] == 1200
    assert row["t_chrom"] == "chr5"
    assert row["t_strand"] == "+"
    assert row["t_src_size"] == 200_000


def test_drops_multi_chrom_group():
    df = _mkdf(
        [
            ("q1", "Mus_musculus", "chr5", 1000, 1100, "+", 200_000),
            ("q1", "Mus_musculus", "chr7", 5000, 5100, "+", 150_000),
        ]
    )
    out = filter_single_chrom_strand(df)
    assert out.is_empty()


def test_drops_multi_strand_group():
    df = _mkdf(
        [
            ("q1", "Mus_musculus", "chr5", 1000, 1100, "+", 200_000),
            ("q1", "Mus_musculus", "chr5", 1100, 1200, "-", 200_000),
        ]
    )
    out = filter_single_chrom_strand(df)
    assert out.is_empty()


def test_keeps_independent_groups_per_query_species():
    df = _mkdf(
        [
            ("q1", "Mus_musculus", "chr5", 1000, 1100, "+", 200_000),
            # Different query, same species, multi-chrom -> dropped
            ("q2", "Mus_musculus", "chr5", 2000, 2100, "+", 200_000),
            ("q2", "Mus_musculus", "chr7", 6000, 6100, "+", 150_000),
            # Same query, different species -> kept independently
            ("q1", "Bos_taurus", "chr3", 500, 700, "-", 100_000),
        ]
    )
    out = filter_single_chrom_strand(df).sort(["query_name", "species"])
    assert out.height == 2
    assert set(out["query_name"]) == {"q1"}
    assert set(out["species"]) == {"Bos_taurus", "Mus_musculus"}


def test_empty_input_returns_empty_with_schema():
    empty = _mkdf([]).clear()
    out = filter_single_chrom_strand(empty)
    assert out.is_empty()
    assert set(out.columns) >= {
        "query_name",
        "species",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
    }


def test_filter_length_inclusive_bounds():
    df = _mkdf(
        [
            ("q1", "Mus_musculus", "chr5", 0, 127, "+", 1_000),  # 127 < 128
            ("q2", "Mus_musculus", "chr5", 0, 128, "+", 1_000),  # 128 OK
            ("q3", "Mus_musculus", "chr5", 0, 512, "+", 1_000),  # 512 OK
            ("q4", "Mus_musculus", "chr5", 0, 513, "+", 1_000),  # 513 > 512
        ]
    )
    out = filter_length(df, min_len=128, max_len=512).sort("query_name")
    assert out["query_name"].to_list() == ["q2", "q3"]


def test_filter_length_default_bounds():
    df = _mkdf(
        [
            ("q1", "Mus_musculus", "chr5", 0, 200, "+", 1_000),
        ]
    )
    out = filter_length(df)
    assert out.height == 1
