"""Section C step 4 — RepeatMasker pass on a sampled subset of v1 sequences.

Selects N rows per species from a representative grid (set via SPECIES below
once Section B/C identifies worst/best offenders by GCF/GCA × quality_source).
Writes FASTA, runs RepeatMasker, parses .out into a per-row mask map, then
compares against v1's existing lowercase per row.

Usage:
    SPECIES="Homo_sapiens Mus_musculus Bos_taurus Ceratotherium_simum ..." \\
        python section_c_repeatmasker.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl


WD = Path.home() / "audit"
OUT = WD / "section_c_rm"
OUT.mkdir(parents=True, exist_ok=True)

SRC = WD / "all_species_with_sequence.parquet"

SPECIES = os.environ.get("SPECIES", "").split()
ROWS_PER_SPECIES = int(os.environ.get("ROWS_PER_SPECIES", "1000"))
SEED = 42


def sample_to_fasta(species: list[str]) -> Path:
    print(f"[RM] Sampling {ROWS_PER_SPECIES} rows × {len(species)} species ...")
    lf = pl.scan_parquet(str(SRC)).filter(pl.col("species").is_in(species))
    df = lf.collect(streaming=True)

    sampled = (
        df.group_by("species")
          .agg(pl.all().shuffle(seed=SEED).head(ROWS_PER_SPECIES))
          .explode(pl.exclude("species"))
    )

    fa_path = OUT / "sample.fa"
    with fa_path.open("w") as f:
        for i, row in enumerate(sampled.iter_rows(named=True)):
            tag = f"row{i:07d}|{row['species']}|{row['query_name']}|{row['t_chrom']}|{row['t_start']}-{row['t_end']}|{row['t_strand']}"
            f.write(f">{tag}\n{row['sequence']}\n")

    sampled.write_parquet(OUT / "sample.parquet")
    print(f"[RM] FASTA: {fa_path}  ({sampled.height} rows)")
    return fa_path


def run_repeatmasker(fa: Path) -> Path:
    out_dir = OUT / "rm_out"
    out_dir.mkdir(exist_ok=True)
    print(f"[RM] Running RepeatMasker (species=mammal, default-engine)...")
    # -species mammals: closest broad library that covers zoonomia leaves;
    # -pa N: parallelism; -dir: output dir; -nolow: skip low-complexity (signal
    #   we want is TE/repeat-mask, not SDUST-style stuff).
    # Drop -nolow to match NCBI's mask more closely if needed.
    nproc = os.cpu_count() or 4
    subprocess.run([
        "RepeatMasker",
        "-species", "mammal",
        "-pa", str(max(1, nproc // 4)),  # RM spawns 4 threads per job
        "-dir", str(out_dir),
        "-xsmall",  # soft-mask in .masked output (lowercase repeats)
        "-no_is",
        str(fa),
    ], check=True)
    print(f"[RM] DONE. Outputs in {out_dir}")
    return out_dir


def parse_rm_out(rm_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """Parse the .out file to per-row list of (start, end) masked intervals (0-based half-open)."""
    out_file = next(rm_dir.glob("*.out"))
    intervals: dict[str, list[tuple[int, int]]] = {}
    with out_file.open() as f:
        # Skip 3 header lines
        for _ in range(3):
            f.readline()
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            tag = parts[4]
            # RepeatMasker uses 1-based inclusive coords; convert to 0-based half-open.
            try:
                s = int(parts[5]) - 1
                e = int(parts[6])
            except ValueError:
                continue
            intervals.setdefault(tag, []).append((s, e))
    return intervals


def compare(fa: Path, rm_dir: Path):
    print("[RM] Computing concordance against v1's existing lowercase...")
    intervals = parse_rm_out(rm_dir)
    sample_df = pl.read_parquet(OUT / "sample.parquet")

    rows = []
    for i, row in enumerate(sample_df.iter_rows(named=True)):
        tag = f"row{i:07d}|{row['species']}|{row['query_name']}|{row['t_chrom']}|{row['t_start']}-{row['t_end']}|{row['t_strand']}"
        seq = row["sequence"]
        seq_len = len(seq)
        # v1 lowercase mask
        v1_mask = [c.islower() for c in seq]
        # RM mask
        rm_mask = [False] * seq_len
        for s, e in intervals.get(tag, []):
            for k in range(max(0, s), min(seq_len, e)):
                rm_mask[k] = True
        # N positions
        n_mask = [c in "Nn" for c in seq]
        non_n_count = seq_len - sum(n_mask)

        v1_lower = sum(1 for j, v in enumerate(v1_mask) if v and not n_mask[j])
        rm_only = sum(1 for j in range(seq_len) if rm_mask[j] and not v1_mask[j] and not n_mask[j])
        v1_only = sum(1 for j in range(seq_len) if v1_mask[j] and not rm_mask[j] and not n_mask[j])
        both = sum(1 for j in range(seq_len) if v1_mask[j] and rm_mask[j] and not n_mask[j])
        rows.append({
            "species": row["species"],
            "query_name": row["query_name"],
            "seq_len": seq_len,
            "non_n_len": non_n_count,
            "v1_lower": v1_lower,
            "rm_total": sum(1 for j in range(seq_len) if rm_mask[j] and not n_mask[j]),
            "rm_only": rm_only,
            "v1_only": v1_only,
            "both": both,
        })

    cmp_df = pl.DataFrame(rows)
    # Per-species summary
    by_species = cmp_df.group_by("species").agg([
        pl.len().alias("n_rows"),
        pl.col("seq_len").sum().alias("total_bases"),
        pl.col("non_n_len").sum().alias("total_non_n"),
        pl.col("v1_lower").sum().alias("total_v1_lower"),
        pl.col("rm_total").sum().alias("total_rm"),
        pl.col("rm_only").sum().alias("total_rm_only"),
        pl.col("v1_only").sum().alias("total_v1_only"),
        pl.col("both").sum().alias("total_both"),
    ])
    by_species = by_species.with_columns([
        (pl.col("total_v1_lower") / pl.col("total_non_n")).alias("v1_lower_frac"),
        (pl.col("total_rm") / pl.col("total_non_n")).alias("rm_frac"),
        # Jaccard = both / (both + rm_only + v1_only)
        (pl.col("total_both") / (pl.col("total_both") + pl.col("total_rm_only") + pl.col("total_v1_only"))).alias("jaccard"),
        # Coverage of v1's calls by RM
        pl.when(pl.col("total_v1_lower") > 0).then(
            pl.col("total_both") / pl.col("total_v1_lower")
        ).otherwise(0.0).alias("v1_covered_by_rm"),
    ])

    by_species.write_csv(OUT / "rm_vs_v1_per_species.tsv", separator="\t")
    print(by_species)
    print(f"[RM] DONE. Wrote {OUT/'rm_vs_v1_per_species.tsv'}.")


def main():
    if not SPECIES:
        sys.exit("set SPECIES=\"sp1 sp2 ...\" (space-separated; quote to keep)")
    fa = sample_to_fasta(SPECIES)
    rm_dir = run_repeatmasker(fa)
    compare(fa, rm_dir)


if __name__ == "__main__":
    main()
