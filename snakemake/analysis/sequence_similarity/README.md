# Sequence Similarity Analysis for Train/Test Leakage Detection

This pipeline analyzes sequence similarity between training and validation splits across different taxonomic scales to understand potential data leakage in genomic language model training.

## Background

When training genomic language models on multi-species datasets, traditional train/test splits (e.g., holding out a chromosome) may not adequately prevent data leakage. Similar sequences can exist:

1. **Within a genome**: Duplicated genes, paralogs, repetitive elements
2. **Across species**: Orthologs, conserved regulatory elements

This is analogous to the problem identified in protein language models like ESM2, which removes training sequences that share >50% identity with validation sequences.

### References

- [ESM2 paper](https://www.science.org/doi/10.1126/science.ade2574): "All train sequences which match a validation sequence with 50% sequence identity are removed from the train set."
- [Detecting and avoiding homology-based data leakage](https://www.biorxiv.org/content/10.1101/2025.01.22.634321v1): Analysis of data leakage in genome-trained sequence models.

## Goals

1. **Quantify leakage**: Measure what percentage of validation sequences have similar sequences in the training set at various identity thresholds.

2. **Compare across taxonomic scales**: Using datasets with increasing phylogenetic diversity (humans → primates → mammals → vertebrates → animals), understand how cross-species similarity contributes to potential leakage.

3. **Inform filtering thresholds**: Determine appropriate identity/coverage thresholds for filtering similar sequences in future dataset creation.

## Implementation

### Approach

We use [MMseqs2 linclust](https://github.com/soedinglab/MMseqs2) for sequence clustering because:

- **Linear time complexity O(N)**: Can handle billions of sequences
- **Scalable**: Designed for massive metagenomic datasets
- **Nucleotide support**: Works with DNA sequences directly

### Pipeline Steps

```
1. Download datasets from HuggingFace
   └── Convert to FASTA format
   └── Canonicalize sequences (optional, for reverse complement handling)

2. Create MMseqs2 database

3. Cluster sequences at multiple identity thresholds (50%, 60%, 70%, 80%, 90%)

4. Analyze cluster composition
   └── Count validation sequences that cluster with training sequences
   └── This represents potential "leakage"

5. Generate summary statistics and visualizations
```

### Reverse Complement Handling

DNA sequences and their reverse complements encode the same information. The pipeline can optionally canonicalize sequences by converting each to the lexicographically smaller of (sequence, reverse_complement), ensuring that a sequence and its RC are treated as identical.

## Installation

### Dependencies

```bash
# MMseqs2 (choose one method)
conda install -c bioconda mmseqs2
# or
brew install mmseqs2      # macOS with Homebrew
# or
apt install mmseqs2       # Ubuntu/Debian

# Python dependencies (if not already installed via bolinas)
pip install datasets polars snakemake matplotlib seaborn
```

### Verify Installation

```bash
mmseqs version
python -c "from datasets import load_dataset; print('OK')"
snakemake --version
```

## Usage

### Quick Start

```bash
cd snakemake/analysis/sequence_similarity

# Dry-run (validate workflow without executing)
snakemake -n --cores 1

# Run sanity check first (recommended)
snakemake sanity_check --cores 8

# Run the full pipeline
snakemake --cores 16
```

### Sanity Check

Before running the full analysis, run the sanity check to verify the pipeline works:

```bash
snakemake sanity_check --cores 8
```

This clusters validation sequences against themselves and verifies:

1. **Reverse complement handling**: With RC augmentation in the dataset, each sequence should cluster with its reverse complement (expect ~50% of sequences to have matches at 100% identity, since RC pairs become identical after canonicalization)

2. **Sliding window overlap**: With 256bp windows and 128bp step, adjacent windows share 128bp (50% overlap). At 100% identity but lower coverage thresholds, these should cluster together.

**Expected output** at 90% identity threshold:
- Most sequences should have at least one match
- Average cluster size > 1

If you see mostly singletons (no matches), something may be wrong with the configuration.

### Configuration

Edit `config/config.yaml` to customize:

```yaml
# Datasets to analyze
datasets:
  - name: humans
    hf_path: bolinas-dna/genomes-v4-genome_set-humans-intervals-v1_256_128
  - name: primates
    hf_path: bolinas-dna/genomes-v4-genome_set-primates-intervals-v1_256_128
  # Add more as needed...

# Clustering parameters
mmseqs2:
  identity_thresholds: [0.5, 0.6, 0.7, 0.8, 0.9]
  coverage: 0.8
  cov_mode: 0  # Both query AND target must have coverage
  threads: 16

# Analysis settings
analysis:
  consider_reverse_complement: true  # Canonicalize sequences
  sequence_column: seq
```

### Running Specific Steps

```bash
# Only download data
snakemake results/data/humans/metadata.parquet --cores 1

# Only run clustering for one dataset/threshold
snakemake results/clustering/humans/clusters_0.5.tsv --cores 16

# Generate plots only (requires prior steps)
snakemake results/plots/leakage_heatmap.png --cores 1
```

## Output

### Directory Structure

```
results/
├── data/
│   └── {dataset}/
│       ├── train.fasta           # Training sequences
│       ├── validation.fasta      # Validation sequences
│       ├── all_sequences.fasta   # Combined for clustering
│       └── metadata.parquet      # Sequence IDs and split labels
├── mmseqs/
│   └── {dataset}/
│       ├── seqDB*                # MMseqs2 sequence database
│       ├── valDB*                # Validation-only database (sanity check)
│       ├── clusters_{identity}/  # Train+val cluster databases
│       └── val_self_clusters_{identity}/  # Val self-clustering (sanity check)
├── sanity_check/
│   └── {dataset}/
│       ├── val_self_clusters_{identity}.tsv  # Validation self-clustering
│       ├── val_self_stats_{identity}.parquet # Statistics per threshold
│       └── summary.parquet       # Sanity check summary
├── clustering/
│   └── {dataset}/
│       └── clusters_{identity}.tsv  # Train+val cluster assignments
├── analysis/
│   ├── {dataset}/
│   │   └── leakage_stats_{identity}.parquet
│   └── leakage_summary.parquet   # Combined statistics
└── plots/
    ├── leakage_heatmap.png       # Heatmap: dataset × threshold
    └── leakage_by_threshold.png  # Line plot: leakage vs threshold
```

### Interpreting Results

The key metric is **leaked_pct**: the percentage of validation sequences that cluster with at least one training sequence at a given identity threshold.

Example interpretation:
- `humans @ 90% identity: 5% leaked` → 5% of validation sequences are near-identical to training
- `primates @ 50% identity: 30% leaked` → 30% of validation sequences share distant homology with primate training sequences

High leakage at strict thresholds (80-90%) suggests near-duplicate sequences that should definitely be filtered. Leakage at looser thresholds (50-60%) indicates evolutionary homology that may or may not be problematic depending on the evaluation task.

## Next Steps

Based on the analysis results, the next phase is to implement filtering in the dataset creation pipeline:

1. After creating train/validation splits, run MMseqs2 clustering
2. Remove training sequences that cluster with validation sequences (ESM2 approach)
3. This preserves the validation set size while reducing leakage

See [Issue #28](https://github.com/Open-Athena/bolinas-dna/issues/28) for the full implementation plan.
