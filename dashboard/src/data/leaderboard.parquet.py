"""Observable Framework data loader: S3 metrics → one tidy parquet.

Pulls per-(method, dataset, subset) PairwiseAccuracy + SE rows from S3 via
``bolinas.pipelines.evals.leaderboard.normalized_rows``, prepends a
``dataset`` column, concatenates across datasets, and writes the resulting
DataFrame as a parquet blob on stdout for the dashboard to read via DuckDB.

v1 emits mendelian_traits only — the data layer is eval-agnostic so the
remaining two datasets (complex_traits, eqtl) are a one-line add when the
corresponding pages ship.
"""

from __future__ import annotations

import sys

import polars as pl

from bolinas.pipelines.evals.leaderboard import normalized_rows

V1_DATASETS: tuple[str, ...] = ("mendelian_traits",)


def main() -> None:
    parts = []
    for dataset in V1_DATASETS:
        df = normalized_rows(dataset).with_columns(dataset=pl.lit(dataset))
        parts.append(df)
    out = pl.concat(parts, how="vertical_relaxed")
    out.write_parquet(sys.stdout.buffer)


if __name__ == "__main__":
    main()
