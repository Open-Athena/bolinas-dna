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
- **MMseqs2**: Installed via conda environment (`workflow/envs/mmseqs2.yaml`), configured in the default profile

You only need [uv](https://docs.astral.sh/uv/) and the project's Python dependencies:

```bash
uv sync
```

### Storage

Pipeline results are stored in S3 (`s3://oa-bolinas/snakemake/analysis/sequence_similarity/`). A default Snakemake profile at `workflow/profiles/default/config.yaml` configures S3 storage, conda, and cores automatically.

You need AWS credentials with S3 access:
- **On EC2**: Attach an IAM role with `AmazonS3FullAccess` to the instance
- **On your laptop**: Run `aws configure` with an IAM user's access key

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
uv run snakemake -n

# Run sanity check first (recommended)
uv run snakemake sanity_check

# Run the full pipeline
uv run snakemake

# Or run just the MMseqs2 analysis:
uv run snakemake mmseqs2_analysis
```

> **Note**: The default profile configures `--use-conda` automatically to install MMseqs2 via conda.
> Conda envs are cached after the first run.

### Sanity Check

Before running the full analysis, run the sanity check to verify the pipeline works:

```bash
uv run snakemake sanity_check
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
uv run snakemake test_masking --resources mem_mb=512
```

#### What the test does

The test generates 6 synthetic 256bp sequences (`results/tests/synthetic.fasta`) and
tests them with both `mmseqs cluster` and `mmseqs search`, with and without lowercase masking:

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

**Cluster test** (all 6 sequences in one database):

| Pair | `--mask-lower-case 0` | `--mask-lower-case 1` |
|------|----------------------|----------------------|
| `genuine_A` / `genuine_B` | Same cluster | Same cluster |
| `repeat_only_A` / `repeat_only_B` | Same cluster | **Different clusters** |
| `mixed_A` / `mixed_B` | Same cluster | Same cluster |

**Search test** (A sequences as query, B sequences as target):

| Pair | `--mask-lower-case 0` | `--mask-lower-case 1` |
|------|----------------------|----------------------|
| `genuine_A` → `genuine_B` | Hit | Hit |
| `repeat_only_A` → `repeat_only_B` | Hit | **No hit** |
| `mixed_A` → `mixed_B` | Hit | Hit |

The key assertion: with masking on, sequences that share **only** a repeat (and have
unrelated flanking regions) should **not** match, while sequences with genuine homology
should still match regardless of masking. This holds for both `cluster` and `search`.

#### Soft-masking and `canonical_sequence()`

The `canonical_sequence()` function in `common.smk` converts each sequence to the
lexicographically smaller of (sequence, reverse complement) so that both strands are treated
identically. This function **preserves case**: it compares strands case-insensitively but
returns the original (mixed-case) sequence. This is critical — if `canonical_sequence()`
called `.upper()`, it would destroy the soft-masking before sequences reach MMseqs2.

#### MMseqs2 gotchas discovered during development

1. **`linclust` silently ignores `--mask-lower-case`.** Only `mmseqs cluster` and
   `mmseqs search` (which use the cascade prefilter + alignment pipeline) honor the flag.
   The pipeline uses `mmseqs cluster` or `mmseqs search` (not `linclust`) everywhere to
   ensure consistent repeat masking.

2. **`--mask-lower-case` must be passed to both `createdb` and `cluster`/`search`.** The
   `createdb` step stores masking metadata in the database; the `cluster`/`search` step
   uses it during k-mer seeding. Passing the flag only to `cluster`/`search` has no effect.

3. **`mmseqs search` requires `--search-type 3` for small nucleotide databases.** MMseqs2
   auto-detects nucleotide vs protein from the database, but with very few sequences (e.g.,
   3) the heuristic can fail. Passing `--search-type 3` (nucleotide) explicitly avoids this.

4. **Simple dinucleotide repeats (e.g., `atatat...`) are not suitable test sequences.**
   They produce only 2 unique k-mers (for any k), which makes them behave unpredictably
   in k-mer-based algorithms. The test uses a random high-complexity lowercase block
   (simulating a realistic transposable element fragment) to isolate `--mask-lower-case`
   behaviour from MMseqs2's built-in compositional bias filtering (`--mask`).

5. **`--mask` (compositional bias / tandem repeat masking) is a separate parameter** from
   `--mask-lower-case`. The former uses an algorithm similar to DUST/SEG to detect and mask
   low-complexity regions at runtime; the latter uses pre-existing case information in the
   input FASTA. Both default to 0 in `linclust` but `--mask` defaults to 1 in `cluster`.

### Configuration

Edit `config/config.yaml` to customize:

```yaml
# Datasets to analyze
datasets:
  # Promoters (intervals-v1)
  - name: humans_promoters
    hf_path: bolinas-dna/genomes-v4-genome_set-humans-intervals-v1_256_128
  - name: primates_promoters
    hf_path: bolinas-dna/genomes-v4-genome_set-primates-intervals-v1_256_128
  - name: mammals_promoters
    hf_path: bolinas-dna/genomes-v4-genome_set-mammals-intervals-v1_256_128
  # CDS (intervals-v5)
  - name: humans_cds
    hf_path: bolinas-dna/genomes-v4-genome_set-humans-intervals-v5_256_128
  - name: primates_cds
    hf_path: bolinas-dna/genomes-v4-genome_set-primates-intervals-v5_256_128
  - name: mammals_cds
    hf_path: bolinas-dna/genomes-v4-genome_set-mammals-intervals-v5_256_128

# MMseqs2 parameters (identity × coverage grid)
mmseqs2:
  identity_thresholds: [0.3, 0.5]
  coverage_thresholds: [0.3, 0.5, 0.7]

# Analysis settings
analysis:
  consider_reverse_complement: true
  sequence_column: seq
```

### Running Specific Steps

```bash
# Only download data
uv run snakemake results/data/humans_promoters/metadata.parquet

# Only run MMseqs2 clustering for one dataset/threshold combo
uv run snakemake results/clustering/humans_promoters/clusters_id0.5_cov0.6.tsv
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
│       └── clusters_id{identity}_cov{coverage}/  # Cluster databases
│
├── sanity_check/                    # Validation self-similarity
│   └── {dataset}/
│       └── summary.parquet
│
├── clustering/                      # MMseqs2 results
│   └── {dataset}/
│       └── clusters_id{identity}_cov{coverage}.tsv
│
├── analysis/                        # MMseqs2 leakage analysis
│   ├── {dataset}/
│   │   └── leakage_stats_id{identity}_cov{coverage}.parquet
│   └── leakage_summary.parquet
│
├── tests/                           # Synthetic masking tests
│   ├── synthetic.fasta              # 6 synthetic sequences
│   ├── clusters_masklc0.tsv         # Clusters without masking
│   ├── clusters_masklc1.tsv         # Clusters with masking
│   ├── masking_test_summary.txt     # Cluster test pass/fail assertions
│   ├── search_query.fasta           # Query sequences (A) for search test
│   ├── search_target.fasta          # Target sequences (B) for search test
│   ├── search_hits_masklc0.tsv      # Search hits without masking
│   ├── search_hits_masklc1.tsv      # Search hits with masking
│   └── search_masking_test_summary.txt  # Search test pass/fail assertions
│
└── plots/
    ├── train_matches_median.svg     # Median train matches heatmap
    └── train_matches_mean.svg       # Mean train matches heatmap
```

### Interpreting Results

The key metric is **leaked_pct**: the percentage of validation sequences that cluster with at least one training sequence at a given identity threshold.

## Results

### Dataset Summary

| Dataset | Interval type | Train seqs | Val seqs | Taxonomic scope |
|---------|--------------|-----------|----------|-----------------|
| humans_promoters | Promoters (v1) | 212,066 | 14,030 | Single genome (Homo sapiens), val = held-out chromosome |
| primates_promoters | Promoters (v1) | 1,814,468 | 14,030 | Multiple primate genomes, val = same human held-out chromosome |
| mammals_promoters | Promoters (v1) | 12,908,076 | 14,030 | 81 mammalian genomes, val = same human held-out chromosome |
| humans_cds | CDS (v5) | 504,572 | 34,680 | Single genome (Homo sapiens), val = held-out chromosome |
| primates_cds | CDS (v5) | 5,640,416 | 34,680 | Multiple primate genomes, val = same human held-out chromosome |
| mammals_cds | CDS (v5) | 41,773,672 | 34,680 | 81 mammalian genomes, val = same human held-out chromosome |

### Repeat Masking Test

The synthetic masking test (`test_masking` target) confirms `--mask-lower-case 1` works correctly
with both `mmseqs cluster` and `mmseqs search`.

**Cluster test:**

| Pair | Without masking | With masking | Status |
|------|----------------|-------------|--------|
| `genuine_A` / `genuine_B` (true homologs) | Same cluster | Same cluster | PASS |
| `repeat_only_A` / `repeat_only_B` (share only repeat) | Same cluster | **Different clusters** | PASS |
| `mixed_A` / `mixed_B` (homologs + repeat) | Same cluster | Same cluster | PASS |

**Search test** (A sequences as query, B sequences as target):

| Pair | Without masking | With masking | Status |
|------|----------------|-------------|--------|
| `genuine_A` → `genuine_B` (true homologs) | Hit | Hit | PASS |
| `repeat_only_A` → `repeat_only_B` (share only repeat) | Hit | **No hit** | PASS |
| `mixed_A` → `mixed_B` (homologs + repeat) | Hit | Hit | PASS |

All 12 assertions passed. This validates that `--mask-lower-case 1` correctly ignores
soft-masked repeats when determining sequence similarity in both clustering and search modes,
while preserving genuine homology signal.

### Sanity Check (Validation Self-Similarity)

Clustering validation sequences against themselves confirms the pipeline works correctly.
One sanity check is run per interval type (promoters and CDS have different validation sets).

**Promoters** (14,030 val sequences):

| Identity | Coverage | Clusters | Singletons | Avg cluster size |
|----------|----------|----------|------------|-----------------|
| 0.3 | 0.3 | 2,728 | 6 (0.0%) | 5.2 |
| 0.5 | 0.3 | 2,728 | 6 (0.0%) | 5.2 |
| 0.3 | 0.5 | 2,774 | 2 (0.0%) | 5.1 |
| 0.5 | 0.5 | 2,774 | 2 (0.0%) | 5.1 |
| 0.3 | 0.7 | 6,372 | 0 (0.0%) | 2.2 |
| 0.5 | 0.7 | 6,372 | 0 (0.0%) | 2.2 |

**CDS** (34,680 val sequences):

| Identity | Coverage | Clusters | Singletons | Avg cluster size |
|----------|----------|----------|------------|-----------------|
| 0.3 | 0.3 | 10,750 | 20 (0.1%) | 3.2 |
| 0.5 | 0.3 | 10,750 | 20 (0.1%) | 3.2 |
| 0.3 | 0.5 | 10,811 | 16 (0.0%) | 3.2 |
| 0.5 | 0.5 | 10,811 | 16 (0.0%) | 3.2 |
| 0.3 | 0.7 | 15,136 | 0 (0.0%) | 2.3 |
| 0.5 | 0.7 | 15,136 | 0 (0.0%) | 2.3 |

At coverage = 0.7, cluster size ~2 reflects that most sequences cluster with exactly their
reverse complement, as expected. At lower coverage, adjacent sliding windows (50% overlap)
also cluster together, producing larger clusters.

### MMseqs2 Leakage Analysis

Clustering sweeps a 2D grid of identity × coverage thresholds. For each validation
sequence, we count how many training sequences share its cluster.

#### Median train matches per validation sequence

**Promoters:**

| Identity \ Coverage | 0.3 | 0.5 | 0.7 |
|---------------------|-----|-----|-----|
| **humans_promoters** | | | |
| 0.3 | 0 | 0 | 0 |
| 0.5 | 0 | 0 | 0 |
| **primates_promoters** | | | |
| 0.3 | 9 | 7 | 4 |
| 0.5 | 9 | 7 | 4 |
| **mammals_promoters** | | | |
| 0.3 | 12 | 9 | 5 |
| 0.5 | 12 | 9 | 5 |

**CDS:**

| Identity \ Coverage | 0.3 | 0.5 | 0.7 |
|---------------------|-----|-----|-----|
| **humans_cds** | | | |
| 0.3 | 0 | 0 | 0 |
| 0.5 | 0 | 0 | 0 |
| **primates_cds** | | | |
| 0.3 | 20 | 20 | 18 |
| 0.5 | 20 | 20 | 18 |
| **mammals_cds** | | | |
| 0.3 | 98 | 83 | 61 |
| 0.5 | 99 | 83 | 61 |

#### Key observations

1. **Single-genome (humans): median is 0 for both promoters and CDS.** Within a single
   genome, most validation sequences have no similar training sequence. For promoters,
   leakage is concentrated in a small minority (mean up to 19, max up to 1,035), likely
   from segmental duplications or unmasked repeats. CDS leakage is even lower (mean ~1,
   max 63–66), consistent with coding sequences being more unique within a genome.

2. **CDS leakage is much higher than promoters at multi-species scales.** At primate scale,
   CDS median matches are 18–20 vs 4–9 for promoters. At mammal scale, the gap widens
   dramatically: CDS median is 61–99 vs 5–12 for promoters. This reflects the strong
   conservation of coding sequences across species — CDS orthologs are more easily detected
   by sequence similarity than promoter regions, which diverge faster.

3. **CDS leakage is less sensitive to coverage threshold.** Promoter median drops sharply
   with coverage (primates: 9 → 7 → 4), but CDS barely changes (primates: 20 → 20 → 18;
   mammals: 98 → 83 → 61). This suggests CDS matches tend to be full-length alignments
   (high coverage), while promoter matches are more often partial alignments.

4. **Coverage is the dominant factor for promoters, not identity.** The median drops from
   12 → 9 → 5 (mammals) and 9 → 7 → 4 (primates) as coverage increases from 0.3 → 0.5
   → 0.7. The difference between id=0.3 and id=0.5 is zero or negligible at every level.

5. **Identity 0.3 and 0.5 give identical results.** MMseqs2's cascade clustering converges
   to the same clusters at both thresholds — there is no additional signal below 0.5
   identity for 256bp DNA sequences. This confirms that the ESM2 (50%) and Chao et al.
   (30%) thresholds are equivalent for short DNA sequences.

6. **Mammals: leakage scales with phylogenetic breadth.** With 81 mammalian genomes in
   training, the median validation sequence has 12 training matches (promoters) or 98
   (CDS) at cov=0.3. Promoter mean is 37–38 with max 1,342–1,494, suggesting outlier
   sequences from unmasked repeats or multi-copy gene families. CDS mean is 151 with max
   3,705, reflecting both strong coding sequence conservation and multi-copy gene families
   across 81 species.

7. **Mammals CDS: highest leakage of all datasets.** The 42M training sequences and
   strong CDS conservation produce the most extreme leakage: every validation threshold
   has a non-zero median (61–99). This means the typical CDS validation sequence has
   dozens to ~100 similar training sequences from orthologous CDS across mammalian genomes.

## Next Steps

1. **Expand to additional taxonomic scales** (vertebrates, animals) to see how leakage
   scales with further phylogenetic distance.

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
