# Enhancer Classification

Binary enhancer classifier. For design decisions, results, and iteration
history see [#96](https://github.com/Open-Athena/bolinas-dna/issues/96).

## Dataset output schema

| Column | Type | Description |
|--------|------|-------------|
| `genome` | str | Species (`"homo_sapiens"` or `"mus_musculus"`) |
| `chrom` | str | Chromosome (Ensembl naming: `"1"`, `"X"`, etc.) |
| `start` | int | 0-based start coordinate |
| `end` | int | End coordinate (exclusive) |
| `strand` | str | `"+"` (forward) or `"-"` (reverse complement) |
| `seq` | str | DNA sequence (255bp) |
| `label` | int | 1 = enhancer, 0 = non-enhancer |

## Code layout

| File | Description |
|------|-------------|
| `src/bolinas/enhancer_classification/dataset.py` | `EnhancerDataset` — PyTorch Dataset loading parquet splits |
| `src/bolinas/enhancer_classification/model.py` | `EnhancerClassifier` — Lightning module |
| `src/bolinas/enhancer_classification/train.py` | CLI training script with argparse |
| `workflow/rules/model.smk` | Snakemake rule calling the train CLI |

## Prerequisites

- AWS credentials configured (EC2 IAM role or `aws configure`)
- **Single GPU required** for training (multi-GPU not supported)
- All available CPU cores are used for data loading (`threads: workflow.cores`)

## Usage

```bash
uv sync --group enhancer-classification
uv run snakemake
```

## Configuration

See `config/config.yaml`.
