# `maf_vs_hal_projection`

One-off benchmark: project a small set of human BED windows onto a few
mammals via two backends and compare. Issue #149 follow-up — picks the
projection backend for the future v1 cross-mammal projection pipeline.

## Why this exists

The followup notes
(`/home/ubuntu/.claude/plans/zoonomia-projection-followup-notes.md`)
committed to a single-copy MAF stream backend on theoretical grounds.
Two facts argue for measuring rather than assuming:

1. The 1.1 TB HAL is needed for v2 sequence extraction (`hal2fasta`);
   adding the 779 GB single-copy MAF on top is a real cost (extra storage,
   extra parsing toolchain).
2. The single-copy filter in MAF (`halSingleCopyRegionsExtract`-style block
   filter) and `halLiftover --noDupes` (position-level filter) target the
   same concept but disagree in the long tail (tandem duplicates, alignment
   block boundaries, alignment-graph edge cases).

This benchmark runs both backends on a small subset, reports raw numbers,
and lets the user pick.

## What this produces

A SkyPilot run on `c6id.12xlarge` (us-east-2, 2.85 TB NVMe) that:

1. Mirrors `447-mammalian-2022v1.hal` and `447-mammalian-2022v1.fix2.single.maf.gz`
   from UCSC to `s3://oa-bolinas/staging/` (idempotent — pulls from S3
   first, falls back to UCSC + uploads).
2. Pulls `min0.20.bed.gz` from
   `s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/bed/`
   (output of PR #152).
3. Samples 10,000 windows on `chr1` (seed 42, deterministic).
4. Runs `scripts/run_maf.py` (single-threaded sequential MAF stream).
5. Runs `scripts/run_hal.py` (parallel `halLiftover --noDupes` per
   species).
6. Emits `comparison.md` via `scripts/compare.py` and uploads everything
   under `s3://oa-bolinas/snakemake/analysis/maf_vs_hal_projection/`.

## How to run

```bash
# From the repo root:
sky launch -c maf-vs-hal -y snakemake/analysis/maf_vs_hal_projection/sky/run.yaml
sky logs maf-vs-hal       # tail; final lines print the comparison.md table
sky down  maf-vs-hal      # only at session end
```

If you want to iterate:

```bash
sky exec maf-vs-hal snakemake/analysis/maf_vs_hal_projection/sky/run.yaml
```

## Backend matrix

| | MAF stream | HAL halLiftover |
|---|---|---|
| Source artifact | `447-mammalian-2022v1.fix2.single.maf.gz` (779 GB) | `447-mammalian-2022v1.hal` (1.1 TB) |
| Tool | `bx-python` (sequential `gzip` parser) | `halLiftover --noDupes` (Cactus binaries) |
| Single-copy filter | Pre-applied in MAF (block-level) | At lift time (`--noDupes`, position-level) |
| Per-species call | One pass yields all species at once | One call per species |
| Coordinate emission | Column-precise (walks alignment) | Column-precise (intrinsic to halLiftover) |

Both backends apply the same downstream:

- `bolinas.projection.filter.filter_single_chrom_strand` (drop multi-chrom /
  multi-strand groups; merge surviving min/max span)
- `bolinas.projection.filter.filter_length` (keep `[128, 512]` bp)
- `bolinas.projection.resize.resize_to_length` (resize to 255 bp around
  midpoint, clamp to chrom.sizes)

## Inputs

- **Windows.** 10,000 random rows from `min0.20.bed.gz`, restricted to
  `chr1`, normalized from Ensembl `1` to UCSC `chr1`.
- **Species (3, spanning evolutionary distance):**
  `Mus_musculus` (Rodentia), `Bos_taurus` (Artiodactyla),
  `Loxodonta_africana` (Proboscidea / afrotherian). All in 447 and 241.
  Originally chose `Tenrec_ecaudatus` for the afrotherian slot but the
  alignment uses `Echinops_telfairi` not `Tenrec_ecaudatus` —
  `Loxodonta_africana` is the cleaner pick anyway (larger, better
  assembled, equally distant).

## Metrics

Per backend × species:

| Metric | What |
|---|---|
| Wall time | Backend-level (extrapolate to 10M × 108) |
| `n` | Records after filter + resize (max one per query × species) |
| Recall | `n / n_windows` |
| Midpoint agreement | Fraction of queries where BOTH backends emit on the same `t_chrom` and midpoints differ by ≤50 bp |
| Output bytes | Per-species Parquet size |

No verdict in `comparison.md` — report-only.

## Out of scope

- v1 of the full ~100-species projection pipeline (separate plan, after
  this benchmark).
- v2 sequence extraction (per-species `hal2fasta` → 2bits → `bedtools
  getfasta`).
- Family-dedup CSV from the 447 leaves (separate one-off).

## Files

```
scripts/sample_windows.py   # 10k chr1 windows, Ensembl→UCSC chrom
scripts/run_maf.py          # MAF stream backend
scripts/run_hal.py          # halLiftover backend (parallel per species)
scripts/compare.py          # raw metrics → comparison.md
sky/run.yaml                # SkyPilot one-shot
```

Library code lives in `src/bolinas/projection/` (tested via
`tests/projection/`); whichever backend the user picks for v1 keeps its
module — the loser gets removed.
