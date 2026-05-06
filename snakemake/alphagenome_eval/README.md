# alphagenome_eval — AlphaGenome baseline on matched-pair eval datasets

PairwiseAccuracy ± binomial SE for [AlphaGenome](https://github.com/google-deepmind/alphagenome)
on the new matched-pair eval datasets
(`bolinas-dna/evals_mendelian_traits` and `bolinas-dna/evals_complex_traits`,
PR #159). Provides issue [#154](https://github.com/Open-Athena/bolinas-dna/issues/154)'s
baseline row for the leaderboards in
[#161](https://github.com/Open-Athena/bolinas-dna/issues/161) and
[#162](https://github.com/Open-Athena/bolinas-dna/issues/162).

## What it does

For each `dataset` in `config["datasets"]`:

1. **Score** every variant via the AlphaGenome API. One forward-strand call per
   variant returns L2_DIFF_LOG1P aggregated scores across 7 assays (ATAC,
   DNASE, CHIP_TF, CHIP_HISTONE, CAGE, PROCAP, RNA_SEQ); each assay yields
   many tracks (one per cell type / tissue), so the per-variant output is a
   wide row with hundreds of track columns.
2. **Aggregate** to a single column by taking the max across all track
   columns: `alphagenome_max_l2`. The full per-track table is preserved on S3
   so the aggregation protocol can change later (e.g. per-assay) without
   re-spending the API budget.
3. **Compute** PairwiseAccuracy ± SE per consequence subset on
   `alphagenome_max_l2`. Same column for both datasets — L2 is
   direction-agnostic and fits both the mendelian and complex-trait protocols.

## Outputs

S3 bucket `s3://oa-bolinas/snakemake/alphagenome_eval/`:

```
results/
├── per_track_l2/{dataset}.parquet    # variant cols + per-track L2 columns
├── scores/{dataset}.parquet          # variant cols + alphagenome_max_l2
└── metrics/{dataset}.parquet         # PairwiseAccuracy ± SE per subset
```

## Conventions

- **Train split only.** Test is held out for the final-eval pass.
- **No reverse-complement averaging.** TraitGym's reference pipeline averages
  forward + reverse strand calls. We skip the RC pass to halve the API budget;
  the metric loss has been small in practice.
- **No edge filtering.** The 1MB sequence context wraps near chromosome ends;
  AlphaGenome handles this internally.

## Setup

`ALPHA_GENOME_API_KEY` env var (request one at
[deepmind.google/alphagenome](https://deepmind.google/science/alphagenome)).
Recommended: export it in `~/.bashrc` so every shell — including the one you
launch SkyPilot from — has it set.

```bash
# In ~/.bashrc:
export ALPHA_GENOME_API_KEY=...
```

The pipeline reads it from the env at the start of `compute_per_track_l2`;
SkyPilot inherits it via the `envs:` block in `sky/run.yaml`.

The official Google `alphagenome` Python client is in the optional
`alphagenome-eval` dep group:

```bash
uv sync --extra alphagenome-eval
```

(SkyPilot's `setup:` does this automatically.)

## Usage

### SkyPilot (recommended)

```bash
# Once per session (or persisted in ~/.bashrc):
export ALPHA_GENOME_API_KEY=...

sky launch snakemake/alphagenome_eval/sky/run.yaml -c alphagenome-eval \
    --env ALPHA_GENOME_API_KEY
```

The launch yaml provisions a small CPU EC2 node (`m6i.large`-class,
us-east-2), runs the full pipeline (~2-3 h end-to-end), and writes outputs to
S3. Tear down with `sky down alphagenome-eval`.

### Local (small subsets only)

```bash
cd snakemake/alphagenome_eval

# Dry-run to inspect the DAG.
uv run snakemake -n

# Real run — will hit the AlphaGenome API for ~12K variants if you don't
# subsample first; only do this if you mean to pay the wallclock.
uv run snakemake
```

## Configuration (`config/config.yaml`)

| Key | Purpose |
| --- | --- |
| `input_hf_prefix` | HF prefix for `f"{prefix}_{dataset}"`. |
| `split` | `train` (test held out). |
| `datasets` | List of dataset names. |
| `num_workers` | Threads in the API ThreadPoolExecutor. Keep ≤ 4. |
| `score_column` | Column name written by `aggregate_max` and consumed by `compute_metrics`. |

The 7 assays, the 1MB sequence length, and `L2_DIFF_LOG1P` aggregation type
are **code constants** in `bolinas.evals.alphagenome`, not config.

## Library

Pipeline rules are thin glue around `bolinas.evals.alphagenome`:

- `score_variants_alphagenome(V, num_workers=4)` — main entry; threads through
  forward-strand `model.score_variant` calls and returns a wide DataFrame.
- `parse_score_response(tidy, scorer_repr_to_assay)` — pure helper converting
  AlphaGenome's `tidy_scores` long-format output to a 1-row wide DataFrame
  with `{assay}_{idx}` column names.
- `make_scorers()` — the 7 `CenterMaskScorer(width=None, L2_DIFF_LOG1P)`
  scorers and their reverse map.

Tests at `tests/evals/test_alphagenome.py` cover the parser without touching
the API; the scorer-construction test is gated on `import alphagenome`.
