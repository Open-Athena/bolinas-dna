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

This pipeline uses [MMseqs2](https://github.com/soedinglab/MMseqs2) `search` to find direct pairwise alignments between validation (query) and training (target) sequences. Key flags:

- **`--mask-lower-case 1`**: excludes soft-masked repeats (from RepeatMasker) from k-mer seeding, so that similarity reflects genuine homology rather than shared transposable elements.
- **`--strand 2`**: searches both forward and reverse complement strands, so reverse complements are detected without needing to pre-canonicalize sequences.
- **`--search-type 3`**: forces nucleotide mode to avoid auto-detection issues with small databases.

### Pipeline Flow

```
1. Download datasets from HuggingFace
   └── Convert to FASTA format
   └── Filter out reverse complement rows (_- suffix)

2. MMseqs2 Search Analysis:
   └── Create separate query (val) and target (train) databases
   └── Search val against train at multiple identity × coverage thresholds
   └── Count direct train hits per val sequence

3. Generate summary statistics and visualizations
```

### Reverse Complement Handling

DNA sequences and their reverse complements encode the same information. The HuggingFace datasets contain both strands (original with `_+` suffix, reverse complement with `_-` suffix). This pipeline:

1. **Filters out `_-` rows at download time** — keeps only the forward strand sequences.
2. **Uses `--strand 2`** in mmseqs2 search — searches both strands automatically, so a val sequence will match a train sequence regardless of which strand was stored.

This is simpler and more correct than the previous approach of canonicalizing sequences (picking the lexicographically smaller of seq/RC), which required careful case-preserving comparisons to avoid destroying soft-masking.

## Installation

### Dependencies

MMseqs2 is managed automatically by Snakemake:
- **MMseqs2**: Installed via conda environment (`workflow/envs/mmseqs2.yaml`), configured in the default profile

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
uv run snakemake -n

# Run sanity check first (recommended)
uv run snakemake sanity_check

# Run the full pipeline
uv run snakemake

# Or run just the search analysis:
uv run snakemake analysis
```

> **Note**: The default profile configures `--use-conda` automatically to install MMseqs2 via conda.
> Conda envs are cached after the first run.

### Sanity Check

Before running the full analysis, run the sanity check to verify the pipeline works:

```bash
uv run snakemake sanity_check
```

This searches validation sequences against themselves (promoters only) and verifies:

1. **Sliding window overlap**: With 256bp windows and 128bp step, adjacent windows share 128bp (50% overlap). At lower coverage thresholds, these should produce hits.

2. **Reverse complement detection**: With `--strand 2`, RC matches are found automatically.

**Expected output**: most validation sequences should have at least one match (from overlapping windows and/or RC hits). If you see mostly zero matches, something may be wrong with the configuration.

### Repeat Masking Test

Genomic sequences contain repetitive elements (transposons, satellites, etc.) that can cause
spurious similarity between unrelated sequences. The input data is expected to be
**pre-masked by RepeatMasker**, where lowercase letters indicate repetitive regions and
uppercase letters indicate non-repetitive ("unique") sequence.

MMseqs2's `--mask-lower-case 1` flag excludes lowercase regions from k-mer matching, so
that search reflects genuine homology rather than shared repeats.

Run the synthetic masking + strand test to verify this works correctly:

```bash
uv run snakemake test_masking --resources mem_mb=512
```

#### What the test does

The test generates 6 synthetic 256bp sequences (`results/tests/synthetic.fasta`) and
tests them with `mmseqs search`, with and without lowercase masking:

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

**Search masking test** (A sequences as query, B sequences as target):

| Pair | `--mask-lower-case 0` | `--mask-lower-case 1` |
|------|----------------------|----------------------|
| `genuine_A` → `genuine_B` (true homologs) | Hit | Hit |
| `repeat_only_A` → `repeat_only_B` (share only repeat) | Hit | **No hit** |
| `mixed_A` → `mixed_B` (homologs + repeat) | Hit | Hit |

The key assertion: with masking on, sequences that share **only** a repeat (and have
unrelated flanking regions) should **not** match, while sequences with genuine homology
should still match regardless of masking.

**Strand test** (validates `--strand 2` finds RC matches):

| Search mode | Query | Target | Expected |
|-------------|-------|--------|----------|
| `--strand 1` (forward only) | seq_A | RC(A) | **No hit** |
| `--strand 2` (both strands) | seq_A | RC(A) | **Hit** |

This validates the core assumption that we can drop sequence canonicalization and rely on
mmseqs2 `--strand 2` to search both strands.

#### MMseqs2 gotchas discovered during development

1. **`linclust` silently ignores `--mask-lower-case`.** Only `mmseqs cluster` and
   `mmseqs search` (which use the cascade prefilter + alignment pipeline) honor the flag.
   The pipeline uses `mmseqs search` (not `linclust`) to ensure consistent repeat masking.

2. **`--mask-lower-case` must be passed to both `createdb` and `search`.** The
   `createdb` step stores masking metadata in the database; the `search` step
   uses it during k-mer seeding. Passing the flag only to `search` has no effect.

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
  sequence_column: seq
```

### Running Specific Steps

```bash
# Only download data
uv run snakemake results/data/humans_promoters/metadata.parquet

# Only run search for one dataset/threshold combo
uv run snakemake results/search/humans_promoters/hits_id0.5_cov0.5.tsv
```

## Output

### Directory Structure

```
results/
├── data/
│   └── {dataset}/
│       ├── train.fasta              # Training sequences (forward strand only)
│       ├── validation.fasta         # Validation sequences (forward strand only)
│       └── metadata.parquet         # Sequence IDs and split labels
│
├── sanity_check/                    # Validation self-similarity (promoters only)
│   └── humans_promoters/
│       ├── hits_id{identity}_cov{coverage}/  # Binary result databases
│       ├── hits_id{identity}_cov{coverage}.tsv
│       ├── stats_id{identity}_cov{coverage}.parquet
│       └── summary.parquet
│
├── search/                          # Leakage analysis
│   ├── {dataset}/
│   │   ├── queryDB*                 # Validation sequence database
│   │   ├── targetDB*                # Training sequence database
│   │   ├── hits_id{identity}_cov{coverage}/  # Binary result databases
│   │   ├── hits_id{identity}_cov{coverage}.tsv  # Pairwise hits
│   │   └── stats_id{identity}_cov{coverage}.parquet
│   └── summary.parquet
│
├── tests/                           # Synthetic tests
│   ├── synthetic.fasta              # 6 synthetic sequences
│   ├── search_query.fasta           # Query sequences (A) for search test
│   ├── search_target.fasta          # Target sequences (B) for search test
│   ├── search_hits_masklc0.tsv      # Search hits without masking
│   ├── search_hits_masklc1.tsv      # Search hits with masking
│   ├── search_masking_test_summary.txt  # Search test pass/fail assertions
│   ├── strand_query.fasta           # Query for strand test
│   ├── strand_target.fasta          # Target (RC) for strand test
│   ├── strand1_hits.tsv             # --strand 1 results (expect empty)
│   ├── strand2_hits.tsv             # --strand 2 results (expect hit)
│   └── strand_test_summary.txt      # Strand test pass/fail assertions
│
└── plots/
    ├── train_matches_median.svg     # Median train matches heatmap
    ├── train_matches_mean.svg       # Mean train matches heatmap
    └── pct_filtered_train.svg       # % train seqs filtered heatmap
```

### Interpreting Results

The key metric is **train_matches**: how many training sequences are direct alignment hits for each validation sequence at a given identity × coverage threshold. This directly answers the question "which train sequences are similar to this val sequence?" — exactly what's needed for filtering.

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
with `mmseqs search`.

**Search masking test** (A sequences as query, B sequences as target):

| Pair | Without masking | With masking | Status |
|------|----------------|-------------|--------|
| `genuine_A` → `genuine_B` (true homologs) | Hit | Hit | PASS |
| `repeat_only_A` → `repeat_only_B` (share only repeat) | Hit | **No hit** | PASS |
| `mixed_A` → `mixed_B` (homologs + repeat) | Hit | Hit | PASS |

**Strand test:**

| Search mode | Expected | Status |
|-------------|----------|--------|
| `--strand 1` (forward only) for RC pair | No hit | PASS |
| `--strand 2` (both strands) for RC pair | Hit | PASS |

All 8 assertions passed. This validates that `--mask-lower-case 1` correctly ignores
soft-masked repeats, and `--strand 2` correctly finds reverse complement matches.

### Leakage Analysis Results

Search reports direct pairwise alignments (val → train), with no transitive closure. For each
validation sequence, we count how many training sequences are direct hits at each identity ×
coverage threshold.

#### Median train matches per validation sequence

**Promoters:**

| Identity \ Coverage | 0.3 | 0.5 | 0.7 |
|---------------------|-----|-----|-----|
| **humans_promoters** | | | |
| 0.3 | 0 | 0 | 0 |
| 0.5 | 0 | 0 | 0 |
| **primates_promoters** | | | |
| 0.3 | 16 | 12 | 6 |
| 0.5 | 16 | 12 | 6 |
| **mammals_promoters** | | | |
| 0.3 | 34 | 20 | 10 |
| 0.5 | 34 | 20 | 10 |

**CDS:**

| Identity \ Coverage | 0.3 | 0.5 | 0.7 |
|---------------------|-----|-----|-----|
| **humans_cds** | | | |
| 0.3 | 0 | 0 | 0 |
| 0.5 | 0 | 0 | 0 |
| **primates_cds** | | | |
| 0.3 | 36 | 26 | 20 |
| 0.5 | 36 | 26 | 20 |
| **mammals_cds** | | | |
| 0.3 | 206 | 154 | 108 |
| 0.5 | 206 | 154 | 108 |

#### Key observations

1. **Single-genome (humans): median is 0 for both promoters and CDS.** Within a single
   genome, most validation sequences have no similar training sequence. Leakage is
   concentrated in a small minority, likely from segmental duplications or unmasked repeats.

2. **CDS leakage is much higher than promoters at multi-species scales.** At primate scale,
   CDS median matches are 20–36 vs 6–16 for promoters. At mammal scale, the gap widens
   dramatically: CDS median is 108–206 vs 10–34 for promoters. This reflects the strong
   conservation of coding sequences across species — CDS orthologs are more easily detected
   by sequence similarity than promoter regions, which diverge faster.

3. **CDS leakage is less sensitive to coverage threshold.** Promoter median drops sharply
   with coverage (mammals: 34 → 20 → 10), but CDS changes less dramatically relative to
   its baseline (mammals: 206 → 154 → 108). This suggests CDS matches tend to be
   full-length alignments (high coverage), while promoter matches are more often partial.

4. **Coverage is the dominant factor for promoters, not identity.** The median drops
   substantially as coverage increases from 0.3 → 0.5 → 0.7. The difference between
   id=0.3 and id=0.5 is zero at every level.

5. **Identity 0.3 and 0.5 give identical results.** There is no additional signal below
   50% identity for 256bp DNA sequences. This confirms that the ESM2 (50%) and Chao et al.
   (30%) thresholds are equivalent for short DNA sequences.

6. **Mammals CDS: highest leakage of all datasets.** The 42M training sequences and
   strong CDS conservation produce the most extreme leakage: every threshold has a non-zero
   median (108–206). The typical CDS validation sequence has ~100–200 direct training
   homologs from orthologous CDS across 81 mammalian genomes.

#### % training sequences filtered

The complementary metric: what percentage of the training dataset would be removed if we
filtered all train sequences with at least one val hit at each threshold.

**Promoters:**

| Identity \ Coverage | 0.3 | 0.5 | 0.7 |
|---------------------|-----|-----|-----|
| **humans_promoters** | | | |
| 0.3 | 0.17% | 0.11% | 0.07% |
| 0.5 | 0.17% | 0.11% | 0.07% |
| **primates_promoters** | | | |
| 0.3 | 3.21% | 2.97% | 2.54% |
| 0.5 | 3.21% | 2.97% | 2.54% |
| **mammals_promoters** | | | |
| 0.3 | 1.97% | 1.72% | 1.29% |
| 0.5 | 1.97% | 1.72% | 1.29% |

**CDS:**

| Identity \ Coverage | 0.3 | 0.5 | 0.7 |
|---------------------|-----|-----|-----|
| **humans_cds** | | | |
| 0.3 | 1.48% | 1.05% | 0.67% |
| 0.5 | 1.48% | 1.05% | 0.67% |
| **primates_cds** | | | |
| 0.3 | 6.44% | 5.90% | 5.03% |
| 0.5 | 6.44% | 5.90% | 5.03% |
| **mammals_cds** | | | |
| 0.3 | 5.20% | 4.47% | 3.37% |
| 0.5 | 5.20% | 4.47% | 3.37% |

#### Key observations (% filtered)

1. **Filtering cost is modest across all datasets.** Even at the loosest threshold (id=0.3,
   cov=0.3), the maximum training data loss is 6.44% (primates CDS). For mammals CDS — the
   dataset with the highest per-val-sequence leakage — only 5.20% of the 42M training
   sequences would be filtered.

2. **Primates CDS has the highest % filtered (6.44%), not mammals CDS (5.20%).** Despite
   mammals CDS having far more matches *per val sequence* (median 206 vs 36), the mammals
   training set is 7.4× larger (42M vs 5.6M), so the fraction filtered is actually lower.
   The leakage is spread across more species, but each species contributes a smaller fraction
   of the total training data.

3. **Promoter % filtered is lower than CDS at multi-species scales.** Primates promoters:
   3.21% vs 6.44% for CDS. Mammals promoters: 1.97% vs 5.20% for CDS. This mirrors the
   per-val-sequence pattern — CDS sequences are more conserved across species.

4. **Single-genome filtering is negligible.** Humans promoters: 0.17%, humans CDS: 1.48%.
   Within a single genome, very few training sequences match validation sequences.

## Recommendations for Training Dataset Deduplication

### Threshold recommendations

- **Identity: 0.5** is sufficient. The results show 0.3 and 0.5 produce identical matches
  across all datasets — there is no additional signal below 50% identity for 256bp DNA
  sequences. Using 0.5 is simpler to justify and consistent with the ESM2 threshold.
- **Coverage**: this is the main trade-off. Lower coverage catches more partial homologs
  (e.g. shared exons within a larger window) but risks false positives. The choice depends
  on how aggressively you want to filter.

### Filtering workflow

1. Run `mmseqs search` with val as query, train as target
2. Collect train IDs with hits
3. Remove them from the training set

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
