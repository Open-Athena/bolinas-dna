# evals_v2 — gLM evaluation on matched-pair datasets

PairwiseAccuracy ± binomial SE on the new matched-pair eval datasets
(`bolinas-dna/evals_mendelian_traits` and `bolinas-dna/evals_complex_traits`,
PR #159). Stripped-down successor to `evals_v1`: one metric, one split, no
plotting, GCS-stored checkpoints.

## What it does

For each `model` × `dataset` in the config:

1. **Download** the genome reference (GRCh38, Ensembl 113) and the model
   checkpoint dir from GCS.
2. **Score** every variant with `compute_variant_scores` (LLR + embedding
   distances).
3. **Compute** PairwiseAccuracy ± SE per consequence subset on the
   dataset-appropriate score column:
   - `mendelian_traits` → `minus_llr` (pathogenic > benign)
   - `complex_traits` → `abs_llr` (magnitude)

Outputs land in S3 at `s3://oa-bolinas/snakemake/analysis/evals_v2/results/`:

```
results/
├── genome.fa.gz
├── checkpoints/{model}/                         # cached HF model dir
├── scores/{model}/{dataset}.parquet             # variant cols + LLR/embedding scores
└── metrics/{model}/{dataset}.parquet            # PairwiseAccuracy ± SE per subset
```

## Conventions

- **Train split only.** Test is held out for the final-eval pass; train is
  the development split.
- **Two context conventions are supported.** Per-model `window_size` config
  field selects the number of DNA bases extracted (255 or 256). The
  tokenizer loaded from each checkpoint handles BOS itself.
  - 255 = BOS-using runs (e.g. `exp136-proj_v30`).
  - 256 = no-BOS runs (e.g. `exp55/58/59`).

## Setup

GPU node (a small EC2 GPU is fine — these are ~0.6B-param models). On the
node:

```bash
# Auth for GCS checkpoint pulls.
gcloud auth application-default login

# Verify gcloud is on PATH and AWS creds reach S3 (same setup as
# snakemake/evals/ — see that pipeline's README).
gcloud storage ls gs://marin-us-central1/checkpoints/ | head
aws s3 ls s3://oa-bolinas/snakemake/analysis/evals_v2/ 2>&1 | head
```

## Usage

```bash
cd snakemake/analysis/evals_v2

# Dry-run to inspect the DAG.
uv run snakemake -n

# Run.
uv run snakemake
```

The default profile (`workflow/profiles/default/config.yaml`) uses S3 storage
at `s3://oa-bolinas/snakemake/analysis/evals_v2/`.

## Configuration (`config/config.yaml`)

| Key | Purpose |
| --- | --- |
| `input_hf_prefix` | HF prefix for `f"{prefix}_{dataset.name}"`. |
| `genome_url` | GRCh38 reference. |
| `split` | `train` (or `test` once held-out eval is unlocked). |
| `datasets` | List of `{name, score_column}`. |
| `models` | List of `{name, gcs_path, window_size}`. `gcs_path` includes the `/hf/step-{N}` suffix. |
| `inference.*` | Batch size, workers, `data_transform_on_the_fly`, `torch_compile`. |

## Library

Pipeline rules are thin glue around:

- `bolinas.evals.inference.compute_variant_scores` — model + genome → score columns.
- `bolinas.evals.metrics.compute_pairwise_metrics` — score column → PairwiseAccuracy ± SE per subset.

No new library code is added by this pipeline; both functions are tested at
`tests/evals/test_pairwise_accuracy.py` and elsewhere.
