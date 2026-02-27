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
4. **Materialize eval harness sequences** *(optional)* - For SNV datasets, extracts DNA sequences from the reference genome in eval harness format (`context` / `ref_completion` / `alt_completion`). Triggered automatically for datasets whose name matches `{dataset}_harness_{window_size}` (e.g. `traitgym_mendelian_v2_harness_256`).
5. **Upload** - Uploads processed datasets to HuggingFace Hub

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

- **`genome_url`** - URL to the reference genome FASTA (gzipped). Used to materialize DNA sequences for eval harness datasets. Defaults to GRCh38 primary assembly from Ensembl.

- **`datasets`** - List of datasets to process. Standard dataset names (e.g. `traitgym_mendelian`) go through the annotation and chromosome-split pipeline. To also materialize sequences into eval harness format, append `_harness_{window_size}` to the base dataset name (e.g. `traitgym_mendelian_v2_harness_256` materializes `traitgym_mendelian_v2` with a 256 bp window centered on each variant).
  - Currently supported base datasets:
    - `traitgym_mendelian` - Mendelian trait variants
    - `traitgym_complex` - Complex trait variants
    - `traitgym_mendelian_v2` - Mendelian trait variants (v2)
    - `clinvar_missense` - ClinVar missense variants
    - `gnomad_pls_v1` / `gnomad_pls_v2` - gnomAD pharmacogenomics variants
    - `gwas_coding` - GWAS coding variants

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
- `gonzalobenegas/bolinas_evals-traitgym_mendelian_v2_harness_256` *(eval harness variant with 256 bp windows)*

Local files are stored in `results/dataset/{dataset}/` with train.parquet and test.parquet splits.

### Eval harness dataset columns

Datasets materialized with `_harness_{window_size}` contain additional columns:

| Column | Description |
|---|---|
| `context` | Left flank sequence up to (but not including) the variant position |
| `ref_completion` | Reference allele + right flank sequence |
| `alt_completion` | Alternate allele + right flank sequence |
| `target` | Binary label (renamed from `label`) |

The window is centered on the variant: `context` has length `window_size // 2` and `ref_completion` / `alt_completion` each have length `window_size // 2`.
