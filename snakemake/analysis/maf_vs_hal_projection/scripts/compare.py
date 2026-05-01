"""Compare MAF and HAL backend outputs and emit a markdown report.

Reads per-species Parquets from both backends plus their ``timing.json``
files, computes raw metrics (recall, single-hit fraction, midpoint
agreement, output bytes), and writes ``comparison.md``. Prints the
table to stdout for ``sky logs``.

No verdict — the user reads the table and decides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def _load_species(parquet_dir: Path, species: list[str]) -> dict[str, pl.DataFrame]:
    return {sp: pl.read_parquet(parquet_dir / f"{sp}.parquet") for sp in species}


def _midpoint_agreement(
    a: pl.DataFrame, b: pl.DataFrame, *, tolerance: int = 50
) -> tuple[int, int, float]:
    """Count queries on which both backends emit a result, and how many
    have midpoints within ``tolerance`` bp of each other on the same chrom.

    Returns ``(n_both_emit, n_agree, frac_agree)``.
    """
    if a.is_empty() and b.is_empty():
        return 0, 0, 0.0
    a_mid = a.with_columns(
        midpoint_a=(pl.col("t_start") + pl.col("t_end")) // 2
    ).select(["query_name", "t_chrom", "midpoint_a"])
    b_mid = b.with_columns(
        midpoint_b=(pl.col("t_start") + pl.col("t_end")) // 2
    ).select(["query_name", "t_chrom", "midpoint_b"])
    joined = a_mid.join(b_mid, on=["query_name", "t_chrom"], how="inner")
    n_both = joined.height
    if n_both == 0:
        return 0, 0, 0.0
    diff = (joined["midpoint_a"] - joined["midpoint_b"]).abs()
    n_agree = int((diff <= tolerance).sum())
    return n_both, n_agree, n_agree / n_both


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--maf-dir", type=Path, required=True)
    p.add_argument("--hal-dir", type=Path, required=True)
    p.add_argument("--species", nargs="+", required=True)
    p.add_argument(
        "--n-windows",
        type=int,
        required=True,
        help="benchmark input size for recall denominator",
    )
    p.add_argument(
        "--tolerance", type=int, default=50, help="bp tolerance for midpoint agreement"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    maf = _load_species(args.maf_dir, args.species)
    hal = _load_species(args.hal_dir, args.species)
    maf_timing = json.loads((args.maf_dir / "timing.json").read_text())
    hal_timing = json.loads((args.hal_dir / "timing.json").read_text())

    rows: list[dict[str, object]] = []
    for sp in args.species:
        m, h = maf[sp], hal[sp]
        n_both, n_agree, frac = _midpoint_agreement(m, h, tolerance=args.tolerance)
        m_bytes = (args.maf_dir / f"{sp}.parquet").stat().st_size
        h_bytes = (args.hal_dir / f"{sp}.parquet").stat().st_size
        rows.append(
            {
                "species": sp,
                "maf_n": m.height,
                "hal_n": h.height,
                "maf_recall": m.height / args.n_windows,
                "hal_recall": h.height / args.n_windows,
                "n_both_emit": n_both,
                "n_midpoint_agree": n_agree,
                "midpoint_agree_frac": frac,
                "maf_bytes": m_bytes,
                "hal_bytes": h_bytes,
            }
        )

    md = ["# MAF vs HAL projection benchmark — raw results\n"]
    md.append(f"- Anchor chrom: `{maf_timing['anchor_chrom']}`")
    md.append(f"- Benchmark windows: **{args.n_windows:,}**")
    md.append(f"- Midpoint agreement tolerance: **{args.tolerance} bp**\n")
    md.append("## Wall time\n")
    md.append("| Backend | Total wall (s) |")
    md.append("|---|---:|")
    md.append(f"| MAF stream | {maf_timing['stream_wall_seconds']:,.1f} |")
    md.append(
        f"| HAL (parallel × {hal_timing['threads']}) | "
        f"{hal_timing['wall_seconds_overall']:,.1f} |"
    )
    md.append("")
    md.append("### Per-species halLiftover wall\n")
    md.append("| Species | halLiftover wall (s) |")
    md.append("|---|---:|")
    for sp in args.species:
        md.append(f"| {sp} | {hal_timing['per_species'][sp]['halliftover_wall']:.1f} |")
    md.append("")

    md.append("## Per-species results\n")
    md.append(
        "| Species | MAF n | HAL n | MAF recall | HAL recall | "
        "n both | midpoint agree | agree % | MAF bytes | HAL bytes |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['species']} | {r['maf_n']:,} | {r['hal_n']:,} | "
            f"{r['maf_recall']:.3f} | {r['hal_recall']:.3f} | "
            f"{r['n_both_emit']:,} | {r['n_midpoint_agree']:,} | "
            f"{r['midpoint_agree_frac']:.1%} | {r['maf_bytes']:,} | {r['hal_bytes']:,} |"
        )
    md.append("")

    md.append("## Notes\n")
    md.append(
        "- `MAF n` / `HAL n` are post-filter, post-resize record counts (one row per "
        "(query, species))."
    )
    md.append(
        "- Recall denominator is the number of benchmark windows. A query can "
        "have at most one record per species after `filter_single_chrom_strand`."
    )
    md.append(
        "- `midpoint agree %` is over queries where BOTH backends emit a record "
        f"on the SAME target chrom (within ±{args.tolerance} bp midpoint)."
    )
    md.append(
        "- Disagreements come from: (a) MAF block boundaries, (b) different "
        "single-copy filter semantics (block-level pre-filter in MAF vs "
        "position-level `--noDupes` in halLiftover)."
    )

    args.output.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
