"""Project genomic intervals across genomes via minimap2 alignment."""

from __future__ import annotations

from pathlib import Path

import polars as pl

PAF_SCHEMA: dict[str, pl.DataType] = {
    "query": pl.Utf8,
    "qlen": pl.Int64,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "strand": pl.Utf8,
    "chrom": pl.Utf8,
    "tlen": pl.Int64,
    "start": pl.Int64,
    "end": pl.Int64,
    "matches": pl.Int64,
    "alnlen": pl.Int64,
    "mapq": pl.Int64,
}


def parse_paf(paf_path: str | Path) -> pl.DataFrame:
    """Parse a minimap2 PAF file into a polars DataFrame.

    Keeps only the first 12 fixed PAF fields (ignores optional SAM-style
    tags). Coordinates are 0-based half-open; `chrom`, `start`, `end` refer
    to the target (reference) genome.
    """
    rows: list[dict] = []
    with open(paf_path) as f:
        for line in f:
            if not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            rows.append(
                {
                    "query": p[0],
                    "qlen": int(p[1]),
                    "qstart": int(p[2]),
                    "qend": int(p[3]),
                    "strand": p[4],
                    "chrom": p[5],
                    "tlen": int(p[6]),
                    "start": int(p[7]),
                    "end": int(p[8]),
                    "matches": int(p[9]),
                    "alnlen": int(p[10]),
                    "mapq": int(p[11]),
                }
            )
    if not rows:
        return pl.DataFrame(schema=PAF_SCHEMA)
    return pl.DataFrame(rows, schema=PAF_SCHEMA)


def best_hit_per_query(paf_df: pl.DataFrame) -> pl.DataFrame:
    """Keep the highest-`matches` hit per query, one row per query.

    Ties are broken arbitrarily (polars `unique(keep="first")` after a
    descending sort on `matches`).
    """
    if paf_df.height == 0:
        return paf_df
    return (
        paf_df.sort("matches", descending=True)
        .unique(subset=["query"], keep="first")
        .sort(["chrom", "start", "end"])
    )
