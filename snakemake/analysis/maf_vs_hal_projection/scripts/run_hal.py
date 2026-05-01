"""HAL backend for the benchmark.

For each target species, calls ``halLiftover --noDupes`` on the benchmark
window BED, runs ``halStats --chromSizes`` to get a per-species chrom.sizes
TSV, joins it onto the lift records as ``t_src_size``, applies the same
single-chrom/single-strand merge + length filter + resize-to-255 as the
MAF backend, and writes one Parquet per target species.

Per-species calls are independent and run in parallel via a thread pool.
``halLiftover`` is single-threaded internally; running N species in
parallel saturates roughly N cores.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path

import polars as pl

from bolinas.projection.filter import filter_length, filter_single_chrom_strand
from bolinas.projection.hal import (
    attach_src_size,
    parse_halliftover_bed,
    run_halliftover,
)
from bolinas.projection.resize import resize_to_length


SOURCE_SPECIES = "Homo_sapiens"

RESIZED_SCHEMA: dict[str, pl.DataType] = {
    "query_name": pl.Utf8,
    "species": pl.Utf8,
    "t_chrom": pl.Utf8,
    "t_start": pl.Int64,
    "t_end": pl.Int64,
    "t_strand": pl.Utf8,
    "t_src_size": pl.Int64,
}


def _hal_chrom_sizes(hal: Path, species: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        subprocess.run(
            ["halStats", "--chromSizes", species, str(hal)],
            check=True,
            stdout=f,
        )


def _resize_df(df: pl.DataFrame, target_len: int) -> pl.DataFrame:
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


def _project_one_species(
    species: str,
    *,
    hal: Path,
    src_bed: Path,
    work_dir: Path,
    target_len: int,
    min_pre_len: int,
    max_pre_len: int,
    out_dir: Path,
) -> dict[str, float | int]:
    species_work = work_dir / species
    species_work.mkdir(parents=True, exist_ok=True)

    sizes_path = species_work / "chrom.sizes"
    t_sizes = time.perf_counter()
    _hal_chrom_sizes(hal, species, sizes_path)
    sizes_wall = time.perf_counter() - t_sizes

    raw_bed = species_work / "halliftover.bed"
    lift_wall = run_halliftover(
        hal,
        SOURCE_SPECIES,
        src_bed,
        species,
        raw_bed,
        no_dupes=True,
    )

    t_post = time.perf_counter()
    df = parse_halliftover_bed(raw_bed, species=species)
    n_raw = df.height
    df = attach_src_size(df, sizes_path)
    n_after_size = df.height
    df = filter_single_chrom_strand(df)
    n_after_chromstrand = df.height
    df = filter_length(df, min_len=min_pre_len, max_len=max_pre_len)
    n_after_length = df.height
    # Drop projections on chroms shorter than the target length.
    df = df.filter(pl.col("t_src_size") >= target_len)
    resized = _resize_df(df, target_len)
    if not resized.is_empty():
        assert (resized["t_end"] - resized["t_start"] == target_len).all()
        assert (resized["t_start"] >= 0).all()
        assert (resized["t_end"] <= resized["t_src_size"]).all()
    out_dir.mkdir(parents=True, exist_ok=True)
    resized.write_parquet(out_dir / f"{species}.parquet")
    post_wall = time.perf_counter() - t_post

    return {
        "halliftover_wall": lift_wall,
        "halstats_wall": sizes_wall,
        "post_processing_wall": post_wall,
        "n_raw": n_raw,
        "n_after_attach_src_size": n_after_size,
        "n_after_single_chrom_strand": n_after_chromstrand,
        "n_after_length_filter": n_after_length,
        "n_final": resized.height,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hal", type=Path, required=True)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--species", nargs="+", required=True)
    p.add_argument("--target-len", type=int, default=255)
    p.add_argument("--min-pre-resize-len", type=int, default=128)
    p.add_argument("--max-pre-resize-len", type=int, default=512)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Max parallel halLiftover calls (default: len(species))",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    threads = args.threads or len(args.species)

    overall_t0 = time.perf_counter()
    per_species: dict[str, dict[str, float | int]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(
                _project_one_species,
                sp,
                hal=args.hal,
                src_bed=args.windows,
                work_dir=args.work_dir,
                target_len=args.target_len,
                min_pre_len=args.min_pre_resize_len,
                max_pre_len=args.max_pre_resize_len,
                out_dir=args.out_dir,
            ): sp
            for sp in args.species
        }
        for fut in concurrent.futures.as_completed(futures):
            sp = futures[fut]
            stats = fut.result()
            per_species[sp] = stats
            print(
                f"{sp}: halLiftover={stats['halliftover_wall']:.1f}s "
                f"raw={stats['n_raw']:,} → single-chrom={stats['n_after_single_chrom_strand']:,} "
                f"→ length={stats['n_after_length_filter']:,} → final={stats['n_final']:,}"
            )
    overall_wall = time.perf_counter() - overall_t0

    timing = {
        "backend": "hal",
        "wall_seconds_overall": overall_wall,
        "threads": threads,
        "species": args.species,
        "per_species": per_species,
    }
    (args.out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    print(
        f"\nHAL backend wall (parallel): {overall_wall:,.1f}s "
        f"across {len(args.species)} species at threads={threads}"
    )
    print(f"wrote {args.out_dir}/timing.json")


if __name__ == "__main__":
    main()
