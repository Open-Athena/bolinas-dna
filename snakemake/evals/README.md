# Evaluation Datasets Pipeline

This pipeline downloads and processes evaluation datasets for genomic language models, including TraitGym benchmarks, and uploads them to HuggingFace.

## What it does

1. **Download datasets** - Fetches evaluation datasets from HuggingFace (currently TraitGym)
2. **Annotate subsets** - Adds functional annotation categories:
   - Non-coding transcript exon variants
   - 3' UTR variants
   - 5' UTR variants
   - Proximal non-exonic variants
   - Distal non-exonic variants
3. **Split by chromosome** - Creates train/test splits using chromosome-based partitioning
   - Train: Odd chromosomes (1, 3, 5, ..., X)
   - Test: Even chromosomes (2, 4, 6, ..., Y)
4. **Upload** - Uploads processed datasets to HuggingFace Hub

## Setup

Python dependencies are managed by the main project (see `../../README.md` for installation).

Before uploading datasets, authenticate with HuggingFace:

```bash
huggingface-cli login
```

### Storage

Pipeline results are stored in S3 (`s3://oa-bolinas/snakemake/evals/`). A default Snakemake profile at `workflow/profiles/default/config.yaml` configures S3 storage and cores automatically.

You need AWS credentials with S3 access:
- **On EC2**: Attach an IAM role with `AmazonS3FullAccess` to the instance
- **On your laptop**: Run `aws configure` with an IAM user's access key

## Configuration

Edit `config/config.yaml` to customize the pipeline:

### Required Parameters

- **`datasets`** - List of datasets to process
  - Currently supported:
    - `traitgym_mendelian` - Mendelian trait variants
    - `traitgym_complex` - Complex trait variants

- **`output_hf_prefix`** - HuggingFace repository prefix (e.g., "username/evals")

## Usage

```bash
# Edit configuration file: config/config.yaml

# Run pipeline
uv run snakemake
```

To generate datasets without uploading:

```bash
# Only create local parquet files
uv run snakemake results/dataset/traitgym_mendelian/train.parquet results/dataset/traitgym_mendelian/test.parquet
```

## Output

Datasets are uploaded to HuggingFace Hub at the specified `output_hf_prefix`.

Dataset naming format: `{output_hf_prefix}-{dataset}`

Examples:
- `gonzalobenegas/bolinas_evals-traitgym_mendelian`
- `gonzalobenegas/bolinas_evals-traitgym_complex`

Local files are stored in `results/dataset/{dataset}/` with train.parquet and test.parquet splits.
