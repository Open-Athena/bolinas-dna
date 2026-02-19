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

## Implementation

This pipeline uses [MMseqs2 cluster](https://github.com/soedinglab/MMseqs2) for **sensitive sequence clustering with lowercase repeat masking** (`--mask-lower-case 1`), ensuring that soft-masked repeats (from RepeatMasker) are excluded from k-mer seeding so that clustering reflects genuine homology.

### Pipeline Flow

```
1. Download datasets from HuggingFace
   └── Convert to FASTA format
   └── Canonicalize sequences (for reverse complement handling)

2. MMseqs2 Analysis:
   └── Create database (with --mask-lower-case 1)
   └── Cluster at multiple identity thresholds (mmseqs cluster --mask-lower-case 1)
   └── Analyze cluster composition (train/val mixing)

3. Generate summary statistics and visualizations
```

### Reverse Complement Handling

DNA sequences and their reverse complements encode the same information. The pipeline canonicalizes sequences by converting each to the lexicographically smaller of (sequence, reverse_complement), ensuring that a sequence and its RC are treated as identical.

## Installation

### Dependencies

MMseqs2 is managed automatically by Snakemake:
- **MMseqs2**: Installed via conda environment (`workflow/envs/mmseqs2.yaml`) when using `--use-conda`

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

# Run the full pipeline
uv run snakemake --cores all --use-conda

# Or run just the MMseqs2 analysis:
uv run snakemake mmseqs2_analysis --cores all --use-conda
```

> **Note**: The `--use-conda` flag is required to automatically install MMseqs2 via conda.
> Conda envs are cached after the first run.

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
├── tests/                           # Synthetic masking test
│   ├── synthetic.fasta              # 6 synthetic sequences
│   ├── clusters_masklc0.tsv         # Clusters without masking
│   ├── clusters_masklc1.tsv         # Clusters with masking
│   └── masking_test_summary.txt     # Pass/fail assertions
│
└── plots/
    ├── leakage_heatmap.png              # MMseqs2: dataset × threshold
    └── leakage_by_threshold.png         # MMseqs2: line plot
```

### Interpreting Results

The key metric is **leaked_pct**: the percentage of validation sequences that cluster with at least one training sequence at a given identity threshold.

## Results

### Dataset Summary

| Dataset | Train seqs | Val seqs | Taxonomic scope |
|---------|-----------|----------|-----------------|
| humans | 212,066 | 14,030 | Single genome (Homo sapiens), val = held-out chromosome |
| primates | 1,814,468 | 14,030 | Multiple primate genomes, val = same human held-out chromosome |

### Repeat Masking Test

The synthetic masking test (`test_masking` target) confirms `--mask-lower-case 1` works correctly:

| Pair | Without masking | With masking | Status |
|------|----------------|-------------|--------|
| `genuine_A` / `genuine_B` (true homologs) | Same cluster | Same cluster | PASS |
| `repeat_only_A` / `repeat_only_B` (share only repeat) | Same cluster | **Different clusters** | PASS |
| `mixed_A` / `mixed_B` (homologs + repeat) | Same cluster | Same cluster | PASS |

All 6 assertions passed. This validates that `mmseqs cluster --mask-lower-case 1` correctly
ignores soft-masked repeats when determining sequence similarity, while preserving genuine
homology signal.

### Sanity Check (Validation Self-Similarity)

Clustering validation sequences against themselves confirms the pipeline works correctly.
At 90% identity, 100% of validation sequences have at least one match (their reverse
complement), and cluster sizes decrease as thresholds become more stringent:

| Identity | Coverage | Clusters | Singletons | Avg cluster size |
|----------|----------|----------|------------|-----------------|
| 0.3 | 0.3 | 2,728 | 6 (0.0%) | 5.2 |
| 0.5 | 0.6 | 6,324 | 0 (0.0%) | 2.2 |
| 0.7 | 1.0 | 6,954 | 0 (0.0%) | 2.0 |
| 0.9 | 1.0 | 6,988 | 0 (0.0%) | 2.0 |

At coverage = 1.0 (full-length match), cluster size ~2.0 confirms that nearly every sequence
clusters with exactly its reverse complement, as expected.

### MMseqs2 Leakage Analysis

**Key finding: cross-species similarity is the dominant source of leakage.** Within a single
human genome, leakage is modest (0.3–6.4%). But when training includes other primate genomes,
50–68% of human validation sequences have a similar training sequence — this is genuine
homology from conserved genomic regions, not repetitive elements (repeats are masked).

#### Humans (single genome)

| Identity threshold | Leaked val seqs | Leaked % | Mixed clusters |
|-------------------|----------------|----------|----------------|
| 50% | 891 | 6.35% | 199 |
| 60% | 653 | 4.65% | 131 |
| 70% | 527 | 3.76% | 80 |
| 80% | 292 | 2.08% | 48 |
| 90% | 40 | 0.29% | 14 |

Within-genome leakage comes from segmental duplications and paralogs. Even at the ESM2-style
50% threshold, only ~6% of validation sequences cluster with training — the chromosome-based
split is reasonably effective for a single genome.

#### Primates (multi-species)

| Identity threshold | Leaked val seqs | Leaked % | Mixed clusters |
|-------------------|----------------|----------|----------------|
| 50% | 9,512 | 67.80% | 4,479 |
| 60% | 9,366 | 66.76% | 4,399 |
| 70% | 9,334 | 66.53% | 4,385 |
| 80% | 7,482 | 53.33% | 3,566 |
| 90% | 6,988 | 49.81% | 3,419 |

The leakage is dramatically higher: ~67% at 50% identity and still ~50% at 90% identity.
This is expected — orthologous regions across primates are highly conserved. The plateau
between 50–70% identity (all ~67%) suggests most cross-species matches are high-identity
orthologs, not borderline hits.

#### Implications

1. **Chromosome holdout is insufficient for multi-species datasets.** A held-out human
   chromosome will have orthologs in every other primate genome in the training set.

2. **Filtering must operate across species.** Simply removing duplicates within a genome
   is not enough; training sequences from other species that are orthologous to validation
   regions must also be filtered.

3. **The 50–70% identity plateau in primates** means that lowering the threshold below 70%
   captures very few additional leaked sequences. A 70–80% threshold may be a practical
   choice that removes most genuine leakage without being overly aggressive.

### Plots

![Leakage Heatmap](results/plots/leakage_heatmap.png)

![Leakage by Threshold](results/plots/leakage_by_threshold.png)

## Next Steps

1. **Expand to additional taxonomic scales** (mammals, vertebrates, animals) to see how
   leakage scales with phylogenetic distance.

2. **Implement filtering in the dataset creation pipeline**: after creating train/validation
   splits, remove training sequences that exceed similarity thresholds against validation.
   This follows the ESM2/Chao et al. approach: preserve validation set, filter training.

See [Issue #28](https://github.com/Open-Athena/bolinas-dna/issues/28) for the full implementation plan.

## References

1. Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130. https://doi.org/10.1126/science.ade2574

2. Chao, K.-H. et al. (2025). Predicting dynamic expression patterns in budding yeast with a fungal DNA language model. *bioRxiv*. https://doi.org/10.1101/2025.09.19.677475

3. Rafi, A.M. et al. (2025). Detecting and avoiding homology-based data leakage in genome-trained sequence models. *bioRxiv*. https://doi.org/10.1101/2025.01.22.634321

4. Steinegger, M. & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology*, 35(11), 1026-1028.

5. Li, H. (2018). Minimap2: pairwise alignment for nucleotide sequences. *Bioinformatics*, 34(18), 3094-3100.
