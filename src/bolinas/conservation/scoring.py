"""Parse ``bigWigAverageOverBed`` TSV output into a typed Polars frame.

``bigWigAverageOverBed`` (UCSC kentUtils) emits one TSV row per BED entry
with columns:
  name, size, covered, sum, mean0, mean

Documented at https://genome.ucsc.edu/goldenPath/help/bigWig.html (search
"bigWigAverageOverBed"). When the bigWig has no values across an interval,
``covered=0`` and ``mean`` is ``n/a`` (we coerce that to NaN).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

_COLUMNS: tuple[str, ...] = ("name", "size", "covered", "sum", "mean0", "mean")


def parse_bigwig_average_over_bed(tsv_path: str | Path) -> pl.DataFrame:
    """Read a ``bigWigAverageOverBed`` TSV into a Polars frame.

    Output schema:
      name (str), size (i64), covered (i64), sum (f64), mean0 (f64), mean (f64)

    The ``mean`` column is NaN where ``covered == 0`` (no bigWig signal in
    the interval). All asserts are run after parsing: invariants that
    should hold for every row are checked here so a malformed TSV fails
    fast at the parsing boundary.
    """
    df = pl.read_csv(
        tsv_path,
        separator="\t",
        has_header=False,
        new_columns=list(_COLUMNS),
        schema_overrides={
            "name": pl.Utf8,
            "size": pl.Int64,
            "covered": pl.Int64,
            "sum": pl.Float64,
            "mean0": pl.Float64,
            "mean": pl.Float64,
        },
        null_values=["n/a"],
    )

    assert set(df.columns) == set(_COLUMNS), (
        f"unexpected columns from bigWigAverageOverBed: {df.columns}"
    )
    assert (df["size"] > 0).all(), "non-positive size in bigWigAverageOverBed output"
    assert (df["covered"] >= 0).all(), (
        "negative `covered` in bigWigAverageOverBed output"
    )
    assert (df["covered"] <= df["size"]).all(), (
        "`covered` > `size` in bigWigAverageOverBed output"
    )
    return df
