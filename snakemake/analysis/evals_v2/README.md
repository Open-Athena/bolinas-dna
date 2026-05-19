# evals_v2 — gLM evaluation on matched-pair datasets

AUPRC ± cluster-bootstrap SE on the matched-pair eval datasets
(`bolinas-dna/evals_mendelian_traits`, `bolinas-dna/evals_complex_traits`).
Each HF dataset revision is pinned per-dataset in `config.yaml` via
`hf_revision` so bumping the underlying data triggers re-execution
deterministically. Stripped-down successor to `evals_v1`: one metric, one
split, no plotting, GCS- or HF-stored checkpoints.

## What it does

For each `model` × `dataset` in the config:

1. **Download** the model checkpoint dir (from GCS or HF Hub depending on
   the model entry). The genome reference is read directly from S3 by
   pyfaidx (byte-range reads — no full download).
2. **Score** every variant with `compute_variant_scores`. The score
   bundle is per-strand LLR + JSD (`down_jsd_mean` in issue #175 — the
   per-position 4-nuc next-token JSD averaged over downstream positions).
   `inference.rc=true` (default) computes both strands; the scores
   parquet then carries the raw atoms `llr_fwd`, `llr_rc`, `jsd_fwd`,
   `jsd_rc`. The metrics rule derives `_avg`, `minus_llr_*`, and
   `abs_llr_*` from these — no redundant storage. FWD+RC is the
   validated default per #175 conclusion 2.
3. **Compute** AUPRC ± cluster-bootstrap SE per consequence subset, per
   score column. Cluster bootstrap resamples `match_group`s with
   replacement (preserving the matched 1:k structure) so SE reflects
   the actual sampling unit. The dataset-appropriate LLR protocol comes
   from `score_protocol` in the config (`minus_llr` for mendelian,
   `abs_llr` for complex); each is evaluated on FWD, RC, and AVG, as
   is JSD.

Outputs land in S3 at `s3://oa-bolinas/snakemake/analysis/evals_v2/results/`:

```
results/
├── checkpoints/{model}/                         # cached HF model dir
├── scores/{model}/{dataset}.parquet             # variant cols + per-strand score atoms
└── metrics/{model}/{dataset}.parquet            # AUPRC ± bootstrap SE per (subset × score_type)
```

The metrics parquet has columns
`[score_type, subset, value, se, n_groups, n_rows, model, dataset, split]`,
with aggregate rows `_global_` and `_macro_avg_` per `score_type` —
see `bolinas.pipelines.evals.metrics.compute_auprc_metrics` for details.

## Conventions

- **Train split only.** Test is held out for the final-eval pass; train is
  the development split.
- **Three context conventions are supported.** Per-model `window_size`
  config field selects the number of DNA bases extracted. The tokenizer
  loaded from each checkpoint handles BOS itself.
  - 255 = BOS-using runs (e.g. `exp136-proj_v30-step-9999`, `exp166-v0.1-p1B-step-27329`).
  - 256 = no-BOS runs at 256-token context (e.g. `exp55/58/59`).
  - 512 = no-BOS runs trained at 512 bp context (e.g. `exp21` promoter-yolo).
    Pair with a per-model `batch_size:` override to fit on an A10G; the
    global default of 128 is tuned for 256-context.

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

# Install the genome-s3 group so pyfaidx can read the reference from S3.
uv sync --frozen --group genome-s3
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
| `genome_path` | Canonical GRCh38 FASTA. fsspec URI (e.g. `s3://...`) or local path. The S3 path requires `--group genome-s3` at install time. |
| `split` | `train` (or `test` once held-out eval is unlocked). |
| `datasets` | List of `{name, hf_revision, score_protocol}`. `hf_revision` is the pinned HF dataset commit SHA — bumping it triggers re-execution. `score_protocol` ∈ `{minus_llr, abs_llr}`. |
| `models` | List of `{name, window_size, ...}`. Each entry has exactly one of `gcs_path` (full GCS URI incl. `/hf/step-{N}`) or `hf_repo` (HuggingFace Hub repo ID), plus two optional fields: `datasets: [...]` to restrict which `datasets` this checkpoint evaluates on (defaults to all), and `batch_size: N` to override the global `inference.batch_size` for this checkpoint (useful when context size differs from the global default's tuning). |
| `inference.*` | Batch size, workers, `data_transform_on_the_fly`, `torch_compile`; `rc` (also score the reverse-complement strand — doubles inference time); `n_bootstrap` (AUPRC bootstrap iterations per subset × score_type); `bootstrap_seed` (reproducibility seed; bumping triggers metrics re-execution). |

## Library

Pipeline rules are thin glue around:

- `bolinas.pipelines.evals.inference.compute_variant_scores` — model + genome
  → per-strand score atoms (`llr_fwd`, `llr_rc`, `jsd_fwd`, `jsd_rc`).
- `bolinas.pipelines.evals.metrics.compute_auprc_metrics` — score columns
  → AUPRC ± cluster-bootstrap SE per subset (cluster = `match_group`).

Both are tested at `tests/pipelines/evals/test_metrics.py`,
`tests/pipelines/evals/test_inference.py`, and `tests/model/test_scoring.py`.
