# Sequence Similarity Analysis for Train/Test Leakage Detection

This pipeline analyzes sequence similarity between training and validation splits across different taxonomic scales to understand potential data leakage in genomic language model training.

## Background

When training genomic language models on multi-species datasets, traditional train/test splits (e.g., holding out a chromosome) may not adequately prevent data leakage. Similar sequences can exist:

1. **Within a genome**: Duplicated genes, paralogs, repetitive elements
2. **Across species**: Orthologs, conserved regulatory elements

This is a well-known problem in both protein and genomic foundation models.

### Prior Work and References

| Model/Paper | Approach | Thresholds |
|------------|----------|------------|
| **ESM2** (protein) | Remove train seqs matching val | 50% identity |
| **Fungal DNA LM** (Chao et al.) | Minimap2 alignment, remove train seqs | Coverage ≥5% AND Identity ≥30% |
| **hashFrag** | BLAST + union-find clustering | Alignment score threshold |

**Key References:**

- **ESM2**: [Lin et al., Science 2023](https://www.science.org/doi/10.1126/science.ade2574)
  > "All train sequences which match a validation sequence with 50% sequence identity under this search are removed from the train set."

- **Fungal DNA Language Model**: [Chao, Kuan-Hao et al., bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1)
  > "We aligned every training sequence against both the validation and test sets using Minimap2 (v2.28-r1209; assembly-to-assembly mode, '-x asm'). We removed any training sequence for which both coverage ≥ 5% and identity ≥ 30% in any alignment."

- **hashFrag**: [de Boer Lab, bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.01.22.634321v1)
  > Analysis of homology-based data leakage in genome-trained sequence models.

## Goals

1. **Quantify leakage**: Measure what percentage of training/validation sequences have similar sequences across splits at various thresholds.

2. **Compare across taxonomic scales**: Using datasets with increasing phylogenetic diversity (humans → primates → mammals → vertebrates → animals), understand how cross-species similarity contributes to potential leakage.

3. **Inform filtering thresholds**: Determine appropriate identity/coverage thresholds for filtering similar sequences in future dataset creation.

4. **Compare methodologies**: Evaluate both MMseqs2 clustering and minimap2 alignment approaches.

## Implementation

This pipeline implements **two complementary approaches**:

### Approach 1: MMseqs2 Clustering

[MMseqs2 cluster](https://github.com/soedinglab/MMseqs2) provides **sensitive sequence clustering with lowercase repeat masking** (`--mask-lower-case 1`), ensuring that soft-masked repeats (from RepeatMasker) are excluded from k-mer seeding so that clustering reflects genuine homology.

**Pros:**
- Sensitive cascade prefilter + alignment pipeline
- Honors `--mask-lower-case 1` for repeat-aware clustering
- Good for exploratory analysis at multiple thresholds
- Single identity threshold (simpler)

**Cons:**
- Slower than `linclust` (but still fast for typical dataset sizes)
- Clustering-based (transitive relationships)

### Approach 2: Minimap2 Alignment (Chao et al. Methodology)

[Minimap2](https://github.com/lh3/minimap2) provides **pairwise alignment with detailed statistics**.

**Pros:**
- Independent coverage AND identity thresholds
- Alignment coordinates for detailed analysis
- Follows published Chao et al. methodology
- Non-transitive (direct pairwise comparisons)

**Cons:**
- O(N×M) complexity (slower for very large datasets)
- More complex threshold tuning (2D space)

### Methodology Comparison

| Aspect | MMseqs2 cluster | Minimap2 (Chao et al.) |
|--------|------------------|-----------------|
| Algorithm | Cascade prefilter + Smith-Waterman clustering | Pairwise alignment |
| Repeat masking | `--mask-lower-case 1` (honors soft-masking) | N/A |
| Thresholds | Single: identity ≥ X% | Dual: coverage ≥ X% AND identity ≥ Y% |
| Default | 50% identity | 5% coverage, 30% identity |
| Output | Cluster assignments | PAF with coordinates |
| Best for | Repeat-aware exploration at multiple thresholds | Final filtering, published method |

### Pipeline Flow

```
1. Download datasets from HuggingFace
   └── Convert to FASTA format
   └── Canonicalize sequences (for reverse complement handling)

2. MMseqs2 Analysis:
   └── Create database (with --mask-lower-case 1)
   └── Cluster at multiple identity thresholds (mmseqs cluster --mask-lower-case 1)
   └── Analyze cluster composition (train/val mixing)

3. Minimap2 Analysis (Chao et al.):
   └── Align train → validation
   └── Parse PAF, compute coverage & identity
   └── Identify leaked sequences at threshold combinations

4. Generate summary statistics and visualizations
```

### Reverse Complement Handling

DNA sequences and their reverse complements encode the same information. The pipeline canonicalizes sequences by converting each to the lexicographically smaller of (sequence, reverse_complement), ensuring that a sequence and its RC are treated as identical.

## Installation

### Dependencies

External tools (MMseqs2, minimap2) are managed automatically by Snakemake:
- **MMseqs2**: Installed via conda environment (`workflow/envs/mmseqs2.yaml`) when using `--use-conda`
- **minimap2**: Installed via [Snakemake wrapper](https://snakemake-wrappers.readthedocs.io/en/stable/wrappers/minimap2/aligner.html) (auto-downloaded)

You only need [uv](https://docs.astral.sh/uv/) and the project's Python dependencies:

```bash
uv sync
```

### Verify Installation

```bash
uv run python -c "from datasets import load_dataset; print('OK')"
uv run snakemake --version
```

## Usage

### Quick Start

```bash
cd snakemake/analysis/sequence_similarity

# Dry-run (validate workflow without executing)
uv run snakemake -n --cores all

# Run sanity check first (recommended)
uv run snakemake sanity_check --cores all --use-conda

# Run the full pipeline (both MMseqs2 and minimap2)
uv run snakemake --cores all --use-conda

# Or run specific analyses:
uv run snakemake mmseqs2_analysis --cores all --use-conda   # MMseqs2 only
uv run snakemake minimap2_analysis --cores all --use-conda  # Minimap2 only (Chao et al.)
```

> **Note**: The `--use-conda` flag is required to automatically install external tools
> (MMseqs2 via conda, minimap2 via Snakemake wrapper). Conda envs are cached after the
> first run.

### Sanity Check

Before running the full analysis, run the sanity check to verify the pipeline works:

```bash
uv run snakemake sanity_check --cores all --use-conda
```

This clusters validation sequences against themselves and verifies:

1. **Reverse complement handling**: With RC augmentation in the dataset, each sequence should cluster with its reverse complement (expect ~50% of sequences to have matches at 100% identity, since RC pairs become identical after canonicalization)

2. **Sliding window overlap**: With 256bp windows and 128bp step, adjacent windows share 128bp (50% overlap). At 100% identity but lower coverage thresholds, these should cluster together.

**Expected output** at 90% identity threshold:
- Most sequences should have at least one match
- Average cluster size > 1

If you see mostly singletons (no matches), something may be wrong with the configuration.

### Repeat Masking Test

Genomic sequences contain repetitive elements (transposons, satellites, etc.) that can cause
spurious similarity between unrelated sequences. The input data is expected to be
**pre-masked by RepeatMasker**, where lowercase letters indicate repetitive regions and
uppercase letters indicate non-repetitive ("unique") sequence.

MMseqs2's `--mask-lower-case 1` flag excludes lowercase regions from k-mer matching, so
that clustering reflects genuine homology rather than shared repeats.

Run the synthetic masking test to verify this works correctly:

```bash
uv run snakemake test_masking --cores 1 --use-conda --resources mem_mb=512
```

#### What the test does

The test generates 6 synthetic 256bp sequences (`results/tests/synthetic.fasta`) and
clusters them with and without lowercase masking:

| Sequence | Description |
|----------|-------------|
| `genuine_A` | Random uppercase DNA (non-repetitive) |
| `genuine_B` | `genuine_A` with ~10% point mutations |
| `repeat_only_A` | Random uppercase flanks + **lowercase** repeat (~200bp) in middle |
| `repeat_only_B` | **Different** uppercase flanks + **same** lowercase repeat |
| `mixed_A` | Copy of `genuine_A` with a ~50bp lowercase repeat inserted |
| `mixed_B` | Copy of `genuine_B` with the same ~50bp lowercase repeat inserted |

The "repeat" block is a random high-complexity lowercase sequence (simulating a transposable
element fragment), not a simple dinucleotide repeat (see gotchas below).

#### Expected results

| Pair | `--mask-lower-case 0` | `--mask-lower-case 1` |
|------|----------------------|----------------------|
| `genuine_A` / `genuine_B` | Same cluster | Same cluster |
| `repeat_only_A` / `repeat_only_B` | Same cluster | **Different clusters** |
| `mixed_A` / `mixed_B` | Same cluster | Same cluster |

The key assertion: with masking on, sequences that share **only** a repeat (and have
unrelated flanking regions) should **not** cluster, while sequences with genuine homology
should still cluster regardless of masking.

#### Soft-masking and `canonical_sequence()`

The `canonical_sequence()` function in `common.smk` converts each sequence to the
lexicographically smaller of (sequence, reverse complement) so that both strands are treated
identically. This function **preserves case**: it compares strands case-insensitively but
returns the original (mixed-case) sequence. This is critical — if `canonical_sequence()`
called `.upper()`, it would destroy the soft-masking before sequences reach MMseqs2.

#### MMseqs2 gotchas discovered during development

1. **`linclust` silently ignores `--mask-lower-case`.** Only `mmseqs cluster` (which uses
   the cascade prefilter + alignment pipeline) honors the flag. The pipeline uses
   `mmseqs cluster` (not `linclust`) everywhere to ensure consistent repeat masking.

2. **`--mask-lower-case` must be passed to both `createdb` and `cluster`.** The `createdb`
   step stores masking metadata in the database; the `cluster` step uses it during k-mer
   seeding. Passing the flag only to `cluster` has no effect.

3. **Simple dinucleotide repeats (e.g., `atatat...`) are not suitable test sequences.**
   They produce only 2 unique k-mers (for any k), which makes them behave unpredictably
   in k-mer-based algorithms. The test uses a random high-complexity lowercase block
   (simulating a realistic transposable element fragment) to isolate `--mask-lower-case`
   behaviour from MMseqs2's built-in compositional bias filtering (`--mask`).

4. **`--mask` (compositional bias / tandem repeat masking) is a separate parameter** from
   `--mask-lower-case`. The former uses an algorithm similar to DUST/SEG to detect and mask
   low-complexity regions at runtime; the latter uses pre-existing case information in the
   input FASTA. Both default to 0 in `linclust` but `--mask` defaults to 1 in `cluster`.

### Configuration

Edit `config/config.yaml` to customize:

```yaml
# Datasets to analyze
datasets:
  - name: humans
    hf_path: bolinas-dna/genomes-v4-genome_set-humans-intervals-v1_256_128
  - name: primates
    hf_path: bolinas-dna/genomes-v4-genome_set-primates-intervals-v1_256_128

# MMseqs2 parameters
mmseqs2:
  identity_thresholds: [0.5, 0.6, 0.7, 0.8, 0.9]
  coverage: 0.8
  cov_mode: 0  # Both query AND target must have coverage

# Minimap2 parameters (Chao et al. methodology)
minimap2:
  preset: asm  # Assembly-to-assembly mode
  coverage_threshold: 0.05   # 5% (Chao et al. default)
  identity_threshold: 0.30   # 30% (Chao et al. default)
  # Additional thresholds for 2D analysis
  coverage_thresholds: [0.05, 0.10, 0.20, 0.50]
  identity_thresholds: [0.30, 0.50, 0.70, 0.90]

# Analysis settings
analysis:
  consider_reverse_complement: true
  sequence_column: seq
```

### Running Specific Steps

```bash
# Only download data
uv run snakemake results/data/humans/metadata.parquet --cores all

# Only run MMseqs2 clustering for one dataset/threshold
uv run snakemake results/clustering/humans/clusters_0.5.tsv --cores all --use-conda

# Only run minimap2 alignment for one dataset
uv run snakemake results/minimap2/humans/train_vs_val_parsed.parquet --cores all --use-conda

# Generate minimap2 scatter plot
uv run snakemake results/plots/humans_minimap2_scatter.png --cores all
```

## Output

### Directory Structure

```
results/
├── data/
│   └── {dataset}/
│       ├── train.fasta              # Training sequences
│       ├── validation.fasta         # Validation sequences
│       ├── all_sequences.fasta      # Combined for MMseqs2
│       └── metadata.parquet         # Sequence IDs and split labels
│
├── mmseqs/                          # MMseqs2 intermediate files
│   └── {dataset}/
│       ├── seqDB*                   # Sequence database
│       ├── valDB*                   # Validation-only database
│       └── clusters_{identity}/     # Cluster databases
│
├── sanity_check/                    # Validation self-similarity
│   └── {dataset}/
│       └── summary.parquet
│
├── clustering/                      # MMseqs2 results
│   └── {dataset}/
│       └── clusters_{identity}.tsv
│
├── analysis/                        # MMseqs2 leakage analysis
│   ├── {dataset}/
│   │   └── leakage_stats_{identity}.parquet
│   └── leakage_summary.parquet
│
├── minimap2/                        # Minimap2 results (Chao et al.)
│   ├── {dataset}/
│   │   ├── train_vs_val.paf         # Raw alignments
│   │   ├── train_vs_val_parsed.parquet  # Parsed with metrics
│   │   ├── leakage_stats.parquet    # Leakage at default threshold
│   │   ├── threshold_analysis.parquet   # 2D threshold sweep
│   │   └── leaked_train_ids.txt     # IDs to filter
│   └── leakage_summary.parquet
│
├── tests/                           # Synthetic masking test
│   ├── synthetic.fasta              # 6 synthetic sequences
│   ├── clusters_masklc0.tsv         # Clusters without masking
│   ├── clusters_masklc1.tsv         # Clusters with masking
│   └── masking_test_summary.txt     # Pass/fail assertions
│
└── plots/
    ├── leakage_heatmap.png              # MMseqs2: dataset × threshold
    ├── leakage_by_threshold.png         # MMseqs2: line plot
    ├── {dataset}_minimap2_scatter.png   # Coverage vs identity scatter
    └── {dataset}_minimap2_heatmap.png   # 2D threshold heatmap
```

### Interpreting Results

#### MMseqs2 Results

The key metric is **leaked_pct**: the percentage of validation sequences that cluster with at least one training sequence at a given identity threshold.

Example interpretation:
- `humans @ 90% identity: 5% leaked` → 5% of validation sequences are near-identical to training
- `primates @ 50% identity: 30% leaked` → 30% of validation sequences share distant homology

#### Minimap2 Results (Chao et al.)

The key metrics are:
- **leaked_train_pct**: % of training sequences with alignment to validation exceeding thresholds
- **matched_val_pct**: % of validation sequences that have matching training sequences

The scatter plot shows the 2D distribution of (coverage, identity) for all alignments. The "leaked region" (coverage ≥ 5%, identity ≥ 30% by default) is highlighted.

The threshold heatmap shows how leakage varies across different threshold combinations, helping to choose appropriate filtering parameters.

## Next Steps

Based on the analysis results, the next phase is to implement filtering in the dataset creation pipeline:

1. After creating train/validation splits, run minimap2 alignment (train → val)
2. Remove training sequences that exceed coverage AND identity thresholds
3. This follows the ESM2/Chao et al. approach: preserve validation set, filter training

The `leaked_train_ids.txt` file can be used directly for filtering.

See [Issue #28](https://github.com/Open-Athena/bolinas-dna/issues/28) for the full implementation plan.

## References

1. Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130. https://doi.org/10.1126/science.ade2574

2. Chao, K.-H. et al. (2025). Predicting dynamic expression patterns in budding yeast with a fungal DNA language model. *bioRxiv*. https://doi.org/10.1101/2025.09.19.677475

3. Rafi, A.M. et al. (2025). Detecting and avoiding homology-based data leakage in genome-trained sequence models. *bioRxiv*. https://doi.org/10.1101/2025.01.22.634321

4. Steinegger, M. & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology*, 35(11), 1026-1028.

5. Li, H. (2018). Minimap2: pairwise alignment for nucleotide sequences. *Bioinformatics*, 34(18), 3094-3100.
