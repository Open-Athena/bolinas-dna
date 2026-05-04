"""MAF backend for the benchmark.

Streams ``447-mammalian-2022v1.fix2.single.maf.gz``, intersects each block
with the benchmark window set on the anchor chrom, and projects via
``bolinas.projection.maf.project_window_through_block`` for the configured
target species. Aggregates per ``(query_name, species)`` (single-chrom /
single-strand merge), filters by length, resizes to 255 bp around the
midpoint, and writes one Parquet per target species.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

from bolinas.projection.filter import filter_length, filter_single_chrom_strand
from bolinas.projection.maf import (
    ProjectionRecord,
    parse_maf_blocks,
    project_window_through_block,
)
from bolinas.projection.resize import resize_to_length


ANCHOR_SPECIES = (
    "hg38"  # Zoonomia 447 MAF uses UCSC assembly name for the anchor row, not binomial
)

PROJECTION_SCHEMA: dict[str, pl.DataType] = {
    "query_name": pl.Utf8,
    "species": pl.Utf8,
    "t_chrom": pl.Utf8,
    "t_start": pl.Int64,
    "t_end": pl.Int64,
    "t_strand": pl.Utf8,
    "t_src_size": pl.Int64,
    "t_aligned_len": pl.Int64,
}

RESIZED_SCHEMA: dict[str, pl.DataType] = {
    "query_name": pl.Utf8,
    "species": pl.Utf8,
    "t_chrom": pl.Utf8,
    "t_start": pl.Int64,
    "t_end": pl.Int64,
    "t_strand": pl.Utf8,
    "t_src_size": pl.Int64,
}


def _records_to_frame(records: list[ProjectionRecord]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame(schema=PROJECTION_SCHEMA)
    return pl.DataFrame(
        {
            "query_name": [r.query_name for r in records],
            "species": [r.species for r in records],
            "t_chrom": [r.t_chrom for r in records],
            "t_start": [r.t_start for r in records],
            "t_end": [r.t_end for r in records],
            "t_strand": [r.t_strand for r in records],
            "t_src_size": [r.t_src_size for r in records],
            "t_aligned_len": [r.t_aligned_len for r in records],
        },
        schema=PROJECTION_SCHEMA,
    )


def _resize_df(df: pl.DataFrame, target_len: int) -> pl.DataFrame:
    """Apply ``resize_to_length`` row-by-row; build a fresh DataFrame.

    Polars expressions can't call our pure-Python resize helper directly
    without a UDF, and for ~10k rows the Python loop is fine.
    """
    if df.is_empty():
        return pl.DataFrame(schema=RESIZED_SCHEMA)
    out_starts: list[int] = []
    out_ends: list[int] = []
    for row in df.iter_rows(named=True):
        ns, ne = resize_to_length(
            row["t_start"], row["t_end"], target_len, row["t_src_size"]
        )
        out_starts.append(ns)
        out_ends.append(ne)
    return pl.DataFrame(
        {
            "query_name": df["query_name"].to_list(),
            "species": df["species"].to_list(),
            "t_chrom": df["t_chrom"].to_list(),
            "t_start": out_starts,
            "t_end": out_ends,
            "t_strand": df["t_strand"].to_list(),
            "t_src_size": df["t_src_size"].to_list(),
        },
        schema=RESIZED_SCHEMA,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--maf", type=Path, required=True)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument(
        "--species",
        nargs="+",
        required=True,
        help="Zoonomia leaf names to keep in output",
    )
    p.add_argument("--anchor-chrom", default="chr1")
    p.add_argument("--target-len", type=int, default=255)
    p.add_argument("--min-pre-resize-len", type=int, default=128)
    p.add_argument("--max-pre-resize-len", type=int, default=512)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--use-taffy",
        action="store_true",
        default=True,
        help="Use `taffy view -r` for indexed random access (default: True). "
        "Requires <maf>.tai alongside the MAF and `taffy` on PATH.",
    )
    p.add_argument(
        "--no-taffy",
        dest="use_taffy",
        action="store_false",
        help="Force bx-python sequential scan from byte 0.",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Input is BED6 written by sample_windows.py (score and strand columns
    # exist for halLiftover compatibility but are unused here — anchor is
    # always Homo_sapiens forward strand).
    windows = (
        pl.read_csv(
            args.windows,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "name", "score", "strand"],
            schema_overrides={
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "name": pl.Utf8,
                "score": pl.Int64,
                "strand": pl.Utf8,
            },
        )
        .filter(pl.col("chrom") == args.anchor_chrom)
        .sort("start")
        .select(["chrom", "start", "end", "name"])
    )
    assert windows.height > 0, f"no windows on {args.anchor_chrom}"
    print(f"loaded {windows.height} windows on {args.anchor_chrom}")

    windows_list = list(
        zip(
            windows["start"].to_list(),
            windows["end"].to_list(),
            windows["name"].to_list(),
        )
    )
    species_set = set(args.species)

    # Bound the MAF region to query: minimal span covering all windows on
    # the anchor chrom. Passed to taffy via parse_maf_blocks(region=...)
    # for O(region) random-access reads instead of O(file) sequential
    # scans. Even with sequential scan we keep the early-exit logic below
    # as a defense against missing/broken .tai indexes.
    #
    # taffy's --region wants the full row-0 sequence name as in the MAF
    # `s` line, i.e. `hg38.chr1` (species.chrom), not just `chr1`.
    min_window_start: int = int(windows["start"].min())
    max_window_end: int = int(windows["end"].max())
    region = (
        f"{ANCHOR_SPECIES}.{args.anchor_chrom}:{min_window_start}-{max_window_end}"
        if args.use_taffy
        else None
    )
    print(
        f"MAF parser: {'taffy random access' if region else 'bx-python sequential'} "
        f"region={region!r}"
    )

    t0 = time.perf_counter()
    n_blocks = 0
    n_blocks_on_chrom = 0
    per_species: dict[str, list[ProjectionRecord]] = defaultdict(list)

    # Early-exit (sequential mode only): the Zoonomia 447 MAF is anchored
    # on hg38 and sorted by anchor chrom + start. Once we move past the
    # target chrom or past max_window_end, break. With taffy random access
    # the region is already bounded so these only fire on edge cases.
    seen_target_chrom = False
    win_lo = 0
    for block in parse_maf_blocks(args.maf, region=region):
        n_blocks += 1
        if n_blocks % 200_000 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  scanned {n_blocks:,} blocks ({n_blocks_on_chrom:,} on chrom) "
                f"in {elapsed:,.1f}s"
            )
        # Pick any Homo_sapiens row to check the anchor chrom; the MAF has
        # exactly one Homo_sapiens row per block (single-copy filtered).
        human_row = next(
            (r for r in block if r.species == ANCHOR_SPECIES),
            None,
        )
        if human_row is None:
            continue
        if human_row.chrom != args.anchor_chrom:
            if seen_target_chrom:
                elapsed = time.perf_counter() - t0
                print(
                    f"  passed {args.anchor_chrom} (now on "
                    f"{human_row.chrom}); breaking after {n_blocks:,} blocks "
                    f"in {elapsed:,.1f}s"
                )
                break
            continue
        seen_target_chrom = True
        # Positional early-exit: if this block starts past max_window_end,
        # no future block on the same chrom can overlap any window either.
        if human_row.start >= max_window_end:
            elapsed = time.perf_counter() - t0
            print(
                f"  past max window end ({max_window_end:,}) at anchor "
                f"start={human_row.start:,}; breaking after "
                f"{n_blocks:,} blocks in {elapsed:,.1f}s"
            )
            break
        anchor_row = human_row
        n_blocks_on_chrom += 1
        a_start = anchor_row.start
        a_end = anchor_row.start + anchor_row.size

        while win_lo < len(windows_list) and windows_list[win_lo][1] <= a_start:
            win_lo += 1
        i = win_lo
        while i < len(windows_list) and windows_list[i][0] < a_end:
            w_start, w_end, w_name = windows_list[i]
            records = project_window_through_block(
                block,
                query_name=w_name,
                anchor_species=ANCHOR_SPECIES,
                anchor_chrom=args.anchor_chrom,
                anchor_start=w_start,
                anchor_end=w_end,
            )
            for r in records:
                if r.species in species_set:
                    per_species[r.species].append(r)
            i += 1

    wall = time.perf_counter() - t0
    print(
        f"\nMAF stream wall: {wall:,.1f}s "
        f"({n_blocks:,} blocks, {n_blocks_on_chrom:,} on {args.anchor_chrom})"
    )

    per_species_walls: dict[str, dict[str, float | int]] = {}
    for sp in args.species:
        sp_t0 = time.perf_counter()
        df = _records_to_frame(per_species.get(sp, []))
        n_raw = df.height
        df = filter_single_chrom_strand(df)
        n_after_chromstrand = df.height
        df = filter_length(
            df, min_len=args.min_pre_resize_len, max_len=args.max_pre_resize_len
        )
        n_after_length = df.height
        # Drop projections on chroms shorter than the target length (rare edge
        # case; resize_to_length raises in that case).
        df = df.filter(pl.col("t_src_size") >= args.target_len)
        resized = _resize_df(df, args.target_len)
        if not resized.is_empty():
            assert (resized["t_end"] - resized["t_start"] == args.target_len).all()
            assert (resized["t_start"] >= 0).all()
            assert (resized["t_end"] <= resized["t_src_size"]).all()
        out = args.out_dir / f"{sp}.parquet"
        resized.write_parquet(out)
        sp_wall = time.perf_counter() - sp_t0
        per_species_walls[sp] = {
            "n_raw": n_raw,
            "n_after_single_chrom_strand": n_after_chromstrand,
            "n_after_length_filter": n_after_length,
            "n_final": resized.height,
            "post_processing_wall": sp_wall,
        }
        print(
            f"{sp}: raw={n_raw:,} → single-chrom={n_after_chromstrand:,} "
            f"→ length={n_after_length:,} → final={resized.height:,} "
            f"(post {sp_wall:.1f}s)"
        )

    timing = {
        "backend": "maf",
        "stream_wall_seconds": wall,
        "n_blocks_total": n_blocks,
        "n_blocks_on_anchor_chrom": n_blocks_on_chrom,
        "anchor_chrom": args.anchor_chrom,
        "n_windows": windows.height,
        "species": args.species,
        "per_species": per_species_walls,
    }
    (args.out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    print(f"\nwrote {args.out_dir}/timing.json")


if __name__ == "__main__":
    main()
