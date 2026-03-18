# Enhancer Classification Dataset

Creates binary classification datasets (enhancer vs non-enhancer) for training
and evaluation.

## Output schema

| Column | Type | Description |
|--------|------|-------------|
| `genome` | str | Species (`"homo_sapiens"` or `"mus_musculus"`) |
| `chrom` | str | Chromosome (Ensembl naming: `"1"`, `"X"`, etc.) |
| `start` | int | 0-based start coordinate |
| `end` | int | End coordinate (exclusive) |
| `strand` | str | `"+"` (forward) or `"-"` (reverse complement) |
| `seq` | str | DNA sequence (255bp) |
| `label` | int | 1 = enhancer, 0 = non-enhancer |

## Method

**Positives**: ENCODE SCREEN cCREs (Registry V4) filtered to enhancer-like
classes (dELS, pELS), resized to 255bp windows centered on each element's
midpoint. Soft-masked genomes from Ensembl.

**Negatives**: Random 255bp windows sampled per-chromosome via
`bedtools shuffle -chrom`, excluding positives and undefined (N-rich) regions.
1:1 positive-to-negative ratio.

**Splits**: Configured via composable named split configs. Default (split_v1):
human chr19 held out for validation; remaining human and all mouse for training.

**Augmentation**: Training splits include reverse complement sequences
(strand="-"). Non-training splits are subsampled to `max_samples`.

## Prerequisites

- AWS credentials configured (EC2 IAM role or `aws configure`)

## Usage

```bash
# Dry run
uv run snakemake -n

# Run full pipeline
uv run snakemake

# Run a specific step
uv run snakemake results/cre/homo_sapiens/all.parquet
```

## Configuration

See `config/config.yaml` for all parameters:

- `datasets`: versioned dataset configs, each referencing a split config and
  interval type (e.g., ELS, ELS_conserved_20)
- `splits`: named chromosome split configs
- `max_samples`: per-split subsampling caps
- `window_size`, `seed`, `negative_ratio`: core parameters
