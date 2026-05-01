# `zoonomia_projection_dataset`

Conservation-filtered human-genome BED windows for cross-mammal projection.

## What it produces

For each `min_proportion_conserved` cutoff in `config/config.yaml`:

- `results/scored/homo_sapiens/phyloP_447m_windows.parquet` — every 255 bp human window (step 128, autosomes + X + Y, ACGT only) scored against `phyloP_447m`. Schema:
  - `chrom, start, end, name` — 0-based half-open BED
  - `conserved_bases` — count of bases ≥ calibrated threshold
  - `proportion_conserved` — `conserved_bases / 255`, NaN counted as 0
  - `mean_phylop` — mean phyloP over covered bases (raw track)
  - `n_valid_bases` — bases with bigWig signal (excludes NaN)
- `results/bed/homo_sapiens/min{min_p}.bed.gz` — filtered BED, one per cutoff.

The downstream cross-mammal projection pipeline (out of scope here) consumes the filtered BEDs.

## Why

The next experiment (issue #149) trains a mammalian gLM on human-anchored intervals projected onto ~100 mammals via the Zoonomia 447-mammalian Cactus alignment. That projection needs a curated human-window set. This pipeline produces it.

## Window size = 255 bp (not 256)

The model adds a BOS token. A 255 bp BED becomes a 256-token context after tokenization. Same convention as `enhancer_classification`.

## phyloP_447m threshold (calibration)

`phyloP_447m` doesn't have an established threshold like `phyloP_241m` (which we use at 2.27 throughout the repo). We pick the `phyloP_447m` threshold so the genome-wide count of bases passing it equals the count of bases passing `phyloP_241m >= 2.27` — same passing-nucleotide count, different track.

This calibration is a **one-off analysis**, not a pipeline rule. To run it:

```bash
uv run python scripts/calibrate_447m_threshold.py \
    --output results/calibration/calibration.json
# Prints the calibrated threshold; paste it into config.yaml under
# phyloP_447m_threshold and commit.
```

The script downloads both bigWigs and builds per-base histograms over defined (ACGT) regions of autosomes + X + Y, then linearly interpolates the threshold within the bracketing histogram bin. Asserts `abs_relative_error < 0.01`.

## Pipeline (Snakemake)

```
download_genome → genome_to_2bit → chrom_sizes_filtered → undefined_regions
                                                              ↓
                                          make_windows (tile + N-filter)
                                                              ↓
download_bigwig (phyloP_447m)  →  binarize_447m  ─┐
                                  download_bigwig │
                                  (phyloP_447m)   ├→ score_windows → filter_bed
                                                  ┘
```

Two `bigWigAverageOverBed` passes per scoring run:

- binary track → `sum` (= `conserved_bases`), `mean0` (= `proportion_conserved`)
- raw track → `mean` (= `mean_phylop`), `covered` (= `n_valid_bases`)

NaN handling falls out from using `sum` and `mean0` (see [NaN semantics](#nan-semantics) below).

## Run

```bash
# From repo root:
git checkout -b feat/zoonomia-projection-dataset origin/main
cd snakemake/zoonomia_projection_dataset

# (Optional but recommended: run the calibration first; paste threshold into config.)
uv run python scripts/calibrate_447m_threshold.py

# Dry-run (CLAUDE.md mandate before any real run)
uv run snakemake --profile workflow/profiles/default -n

# Full pipeline
uv run snakemake --profile workflow/profiles/default

# Tests for the library code
uv run pytest tests/conservation/
```

## SkyPilot

CPU-only pipeline. Recommended box: `c6id.2xlarge` in `us-east-2` (8 vCPU, 16 GB RAM, 474 GB local NVMe, ~$0.40/hr). NVMe matters — the bigWigs are ~30 GB each and gp3 EBS is ~10× slower.

Tag instances with `project=dna` (cluster lifecycle: launch once, reuse via `sky exec`, `sky down` only at session end).

## NaN semantics

`phyloP_447m` has NaN at positions with no alignment. We want NaN counted as **non-conserved** (0).

Using `bigWigAverageOverBed`'s `sum` and `mean0` columns (not `mean`):

- `sum` = number of conserved bases. NaN positions contribute 0 either because they're absent from the binary bigWig or because they were filled to 0 — same answer.
- `mean0 = sum / size` = proportion conserved over the **full** 255 bp denominator. NaN remains in the denominator and contributes 0 to the numerator → counted as non-conserved.
- `mean = sum / covered` is the column we *don't* want — that one excludes NaN.

Sanity-checked in `tests/conservation/test_scoring.py`.

## Layout

```
config/config.yaml                  # genome URL, window/step, threshold (placeholder), filters
workflow/Snakefile                  # rule orchestration
workflow/profiles/default/          # cores, S3 storage, conda
workflow/envs/bioinformatics.yaml   # bedtools, kentUtils, wiggletools
workflow/rules/
  common.smk                        # imports + sanity checks
  genome.smk                        # download_genome → 2bit → chrom_sizes → N-regions
  download.smk                      # phyloP bigWig
  windows.smk                       # tile + N-filter (one rule)
  score.smk                         # binarize + score
  filter.smk                        # filtered BED per cutoff
scripts/
  calibrate_447m_threshold.py       # one-off; uses src/bolinas/conservation/
```

Library code lives in `src/bolinas/conservation/` (testable; reused by both the calibration script and the score rule).
