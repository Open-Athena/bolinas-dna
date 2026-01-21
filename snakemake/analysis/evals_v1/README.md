# Evaluation Pipeline v1

This pipeline evaluates genomic language models by computing variant effect predictions and comparing them against labeled datasets.

## Overview

The pipeline:
1. Downloads evaluation datasets from HuggingFace (created by `snakemake/evals/`)
2. Downloads genome reference (GRCh38)
3. Runs inference to compute LLR scores and embedding distances for each variant using model checkpoints
4. Computes evaluation metrics (AUPRC, AUROC, Spearman) globally and on annotation subsets
5. Generates comparison plots

## Setup

### Python Dependencies

The pipeline uses the main project's Python environment. If you haven't already installed dependencies:

```bash
cd /path/to/bolinas-dna
uv sync
```

### Configuration

Edit `config/config.yaml` to specify:

1. **Models to evaluate**: Specify training runs with base paths, context size, and checkpoint steps
   ```yaml
   models:
     - name: gpn_promoter
       base_path: /path/to/training/run
       context_size: 512
       steps: [10000, 20000, 50000, 100000]
   ```

2. **Datasets**: Evaluation datasets from HuggingFace
   ```yaml
   datasets:
     - name: traitgym_mendelian
       hf_path: gonzalobenegas/bolinas_evals-traitgym_mendelian
       split: test
       metrics: [AUPRC, AUROC]
   ```

3. **Inference settings**: Performance tuning (doesn't affect output, won't trigger recomputation)
   ```yaml
   inference:
     batch_size: 128
     num_workers: 4
     data_transform_on_the_fly: true
     torch_compile: false  # Enable for faster inference (requires PyTorch 2.0+)
   ```

## Usage

Run the complete pipeline:

```bash
cd snakemake/analysis/evals_v1
uv run snakemake --cores all
```

Run specific targets:

```bash
# Just compute scores for one dataset/model
uv run snakemake results/scores/traitgym_mendelian/gpn_promoter/10000.parquet --cores 4

# Just compute metrics for one dataset/model/step
uv run snakemake results/metrics/traitgym_mendelian/gpn_promoter/10000.parquet --cores 1

# Just generate plot for one model
uv run snakemake results/plots/metrics_vs_step/gpn_promoter.svg --cores 1
```

Dry run to see what will be executed:

```bash
uv run snakemake --dry-run
```

## Output

### Directory Structure

```
results/
├── genome.fa.gz                        # GRCh38 reference genome
├── scores/
│   └── {dataset}/
│       └── {model}/
│           └── {step}.parquet         # Variant scores (LLR + embedding distances)
├── metrics/
│   └── {dataset}/
│       └── {model}/
│           └── {step}.parquet         # Metrics (global + per subset)
└── plots/
    └── metrics_vs_step/
        └── {model}.svg                # Metric progression across training for each model
```

### Scores Files

Parquet files with columns (aligned by row index with source dataset):
- `llr`: Raw log-likelihood ratio
- `minus_llr`: Negated LLR (higher = more deleterious)
- `abs_llr`: Absolute LLR (higher = more impactful)
- `embed_last_l2`: L2 distance between reference and alternate embeddings (last layer)
- `embed_middle_l2`: L2 distance between reference and alternate embeddings (middle layer)

### Metrics Files

Parquet files with columns:
- `metric`: Metric name (AUPRC, AUROC, Spearman)
- `score_type`: Scoring method (minus_llr, abs_llr, embed_last_l2, embed_middle_l2)
- `subset`: Annotation subset or "global"
- `value`: Metric value

Note: `step` and `dataset` are encoded in the file path, not as columns.

### Plots

- **metrics_vs_step/{model}.svg**: Per-model plots showing metric progression across training steps. Each subplot shows a (dataset, subset) combination with lines for each scoring method (minus_llr, abs_llr, embed_last_l2, embed_middle_l2).

## Annotation Subsets

Datasets created by `snakemake/evals/` include a `subset` column with annotation categories:
- `noncoding_transcript_exon`: Non-coding transcript exon variants
- `three_prime_UTR`: 3' UTR variants
- `five_prime_UTR`: 5' UTR variants
- `proximal_nonexonic`: Proximal non-exonic variants (near genes)
- `distal_nonexonic`: Distal non-exonic variants (far from genes)

Metrics are computed both globally (all variants) and separately for each subset.

## Implementation Details

### Code Organization

The pipeline uses a clean separation between Snakemake rules and Python logic:

- **`src/bolinas/evals/`**: Core Python module with type hints and tests
  - `inference.py`: LLR and embedding distance computation using biofoundation
  - `metrics.py`: Metric computation functions
  - `plotting.py`: Plotting utilities

- **`workflow/rules/`**: Thin Snakemake wrappers
  - `inference.smk`: Download datasets and run inference
  - `metrics.smk`: Compute and aggregate metrics
  - `plots.smk`: Generate plots

### Dependencies

- **biofoundation**: LLR and embedding distance inference utilities
- **transformers**: HuggingFace model loading
- **datasets**: HuggingFace dataset loading
- **pandas**: Data manipulation
- **scikit-learn**: Metrics (AUPRC, AUROC)
- **scipy**: Statistics (Spearman correlation)
- **matplotlib**, **seaborn**: Plotting

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```yaml
inference:
  batch_size: 64  # Default is 128
```

### Missing Checkpoints

Verify checkpoint paths in config exist:
```bash
ls /path/to/training/run/step-10000
```

### HuggingFace Authentication

Some datasets may require authentication:
```bash
huggingface-cli login
```

## Extending the Pipeline

### Adding New Datasets

1. Create dataset using `snakemake/evals/` pipeline
2. Add to config:
   ```yaml
   datasets:
     - name: my_dataset
       hf_path: username/bolinas_evals-my_dataset
       split: test
       metrics: [AUPRC]
   ```

### Adding New Metrics

1. Add metric function to `src/bolinas/evals/metrics.py`:
   ```python
   METRIC_FUNCTIONS["MyMetric"] = lambda label, score: my_metric_fn(label, score)
   ```

2. Reference in config:
   ```yaml
   datasets:
     - name: my_dataset
       metrics: [AUPRC, MyMetric]
   ```

### Adding New Plots

1. Implement plotting function in `src/bolinas/evals/plotting.py`
2. Add rule in `workflow/rules/plots.smk`
3. Add output to `rule all` in `Snakefile`
